import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import zipfile
import os
from gensim.models import KeyedVectors
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

print("Loading FastText embeddings for Hindi and Nepali...")
fasttext_hi = KeyedVectors.load_word2vec_format("cc.hi.300.vec.gz", binary=False)
fasttext_ne = KeyedVectors.load_word2vec_format("cc.ne.300.vec.gz", binary=False)

embedding_dim = fasttext_hi.vector_size  

print("Loading train and test datasets...")


print("Loading train and test datasets...")
train_df = pd.read_csv('dataset/train.csv') 
test_df = pd.read_csv('dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str).fillna('')
test_df['tweet'] = test_df['tweet'].astype(str).fillna('')

if 'label' in train_df.columns:
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    val_texts = val_data['tweet'].tolist()
    val_labels = val_data['label'].tolist()
else:
    print("No validation required as test set doesn't have labels.")

train_texts = train_df['tweet'].tolist()
train_labels = train_df['label'].tolist()

class FastTextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.labels = labels
        self.embeddings = [self.get_fasttext_embeddings(text) for text in tqdm(texts, desc="Embedding texts")]

    def get_fasttext_embeddings(self, text):
        words = text.split()
        word_embeddings = []

        for word in words:
            if word in fasttext_hi:
                emb_hi = fasttext_hi[word]
            else:
                emb_hi = np.zeros(embedding_dim)

            if word in fasttext_ne:
                emb_ne = fasttext_ne[word]
            else:
                emb_ne = np.zeros(embedding_dim)

           
            word_embedding = (emb_hi + emb_ne) / 2 if np.any(emb_hi) and np.any(emb_ne) else emb_hi if np.any(emb_hi) else emb_ne
            word_embeddings.append(word_embedding)

       
        return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(embedding_dim)

    def __getitem__(self, idx):
        item = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return item, label
        return item

    def __len__(self):
        return len(self.embeddings)

class FastTextLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(FastTextLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # Add batch dimension
        out = self.fc(lstm_out[:, -1, :])  # Use last LSTM output
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = FastTextLSTM(embedding_dim=embedding_dim, hidden_dim=128, output_dim=2).to(device)

print("Preparing training dataset...")
train_dataset = FastTextDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

def train_model(model, train_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Start training
train_model(lstm_model, train_loader)

print("Generating predictions for test set...")
test_dataset = FastTextDataset(test_df['tweet'].tolist())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

lstm_model.eval()
lstm_predictions = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Predicting on test data"):
        data = data.to(device)
        outputs = lstm_model(data)
        _, preds = torch.max(outputs, 1)
        lstm_predictions.extend(preds.cpu().numpy())

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir = f'outputs/lstm_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

json_file_path = os.path.join(output_dir, f'submission_lstm_{timestamp}.json')
with open(json_file_path, 'w') as f:
    for idx, pred in zip(test_df['index'], lstm_predictions):
        json.dump({"index": int(idx), "prediction": int(pred)}, f)
        f.write('\n')

zip_file_path = f'{output_dir}.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(json_file_path, arcname=os.path.basename(json_file_path))

print(f"LSTM predictions saved to {zip_file_path}")
