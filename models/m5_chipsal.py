# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import os
import json
import zipfile

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_bert_model = AutoModel.from_pretrained("ai4bharat/indic-bert")

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str)
test_df['tweet'] = test_df['tweet'].astype(str)

train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

class IndicBERTDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts)

train_texts = train_data['tweet'].tolist()
train_labels = train_data['label'].tolist()
val_texts = val_data['tweet'].tolist()
val_labels = val_data['label'].tolist()

train_dataset = IndicBERTDataset(train_texts, train_labels)
val_dataset = IndicBERTDataset(val_texts, val_labels)

class LSTM_CNN_IndicBERT(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(LSTM_CNN_IndicBERT, self).__init__()
        self.indic_bert_model = indic_bert_model
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True) 
        self.conv1 = nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  
            bert_outputs = self.indic_bert_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = bert_outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(last_hidden_state)
        cnn_out = torch.relu(self.conv1(lstm_out.permute(0, 2, 1))) 
        pooled = torch.mean(cnn_out, dim=-1) 
        output = self.fc(pooled)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_cnn_model = LSTM_CNN_IndicBERT(hidden_dim=128, num_classes=2).to(device)

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir_name = f"outputs/{timestamp}_IndicBERT_LSTM_CNN"
os.makedirs(output_dir_name, exist_ok=True)

def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')

        # Evaluate on validation set
        evaluate_model(model, val_loader)

def evaluate_model(model, loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

train_model(lstm_cnn_model, train_loader, val_loader)

test_texts = test_df['tweet'].tolist()
test_dataset = IndicBERTDataset(test_texts)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
lstm_cnn_model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = lstm_cnn_model(input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

submission_df = pd.DataFrame({
    'index': test_df['index'],
    'prediction': predictions
})

json_file_path = f'{output_dir_name}/submission_{timestamp}.json'
submission_df.to_json(json_file_path, orient='records', lines=True)


zip_file_path = f'{output_dir_name}_{timestamp}.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(json_file_path, arcname=f'submission_{timestamp}.json')

print(f"Predictions saved to {zip_file_path}")
