
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
import os
import json
import zipfile
import torch
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2).to("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str)
test_df['tweet'] = test_df['tweet'].astype(str)

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts)

# Prepare train dataset
train_texts = train_df['tweet'].tolist()
train_labels = train_df['label'].tolist()  # Assuming 1 is hate, 0 is non-hate
train_dataset = HateSpeechDataset(train_texts, train_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=xlmr_model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
)

print("Starting fine-tuning of XLM-Roberta...")
trainer.train()

test_texts = test_df['tweet'].tolist()
test_dataset = HateSpeechDataset(test_texts)

print("Generating predictions on the test set...")
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)

submission_df = pd.DataFrame({
    'index': test_df['index'],  # Using 'index' from the test data
    'prediction': pred_labels.tolist()
})

submission_df = submission_df.sort_values(by='index').reset_index(drop=True)

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir_name = f"{timestamp}_XLMRoberta_Classifier"

os.makedirs(f'outputs/{output_dir_name}', exist_ok=True)

json_file_path = f'outputs/{output_dir_name}/submission_{output_dir_name}.json'
json_records = submission_df.apply(lambda row: {"index": int(row['index']), "prediction": int(row['prediction'])}, axis=1)
with open(json_file_path, 'w') as json_file:
    for record in json_records:
        json_file.write(json.dumps(record) + '\n')

zip_file_path = f'outputs/{output_dir_name}.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(json_file_path, arcname=f'submission_{output_dir_name}.json')

print(f"Predictions saved to {zip_file_path}")
