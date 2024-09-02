import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import loralib as lora  

# Define constants
MODEL_NAME = 'ai4bharat/indic-bert'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
LOADER_WORKERS = 4
RANK = 8  # LoRA rank
ALPHA = 16  # LoRA alpha

# Load and preprocess the dataset
df = pd.read_csv('/mnt/Enterprise2/anmol/CHiPSAL_COLING/chipsal-datasets/train/task2.csv')

# Preprocessing function
def preprocess_text(text):
    return text.replace("https://t.co/", "").replace("#", "").replace("@", "")

df['tweet'] = df['tweet'].apply(preprocess_text)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = self.data.iloc[index]['tweet']
        label = self.data.iloc[index]['label']
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create Dataloaders
train_dataset = HateSpeechDataset(train_df, tokenizer, MAX_LEN)
test_dataset = HateSpeechDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=LOADER_WORKERS)

# Define the Model class with LoRA integration and an MLP classifier
class LoRAHateSpeechModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(LoRAHateSpeechModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Apply LoRA to the transformer layers
        lora.mark_only_lora_as_trainable(self.base_model, lora.RANK = RANK, lora.ALPHA = ALPHA)

        # Freeze the remaining layers
        for param in self.base_model.parameters():
            if param.requires_grad:
                param.requires_grad = False

        # MLP Classifier
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = base_output.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits

# Instantiate the model
model = LoRAHateSpeechModel(MODEL_NAME)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training function
def train_model(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, f1

# Training loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_model(model, train_loader, optimizer, scheduler)
    print(f'Train loss: {train_loss}')

    accuracy, f1 = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy}, Test F1 Score: {f1}')

# Final evaluation and report
accuracy, f1 = evaluate_model(model, test_loader)
print(f'Final Test Accuracy: {accuracy}')
print(f'Final Test F1 Score: {f1}')
print("Classification Report:")
print(classification_report(test_df['label'], predictions))
