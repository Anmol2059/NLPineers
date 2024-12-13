
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from datetime import datetime
import os
import json
import zipfile
import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/hindi-abusive-MuRIL")
muril_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/hindi-abusive-MuRIL", num_labels=2).to("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str)
test_df['tweet'] = test_df['tweet'].astype(str)

train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

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

train_texts = train_data['tweet'].tolist()
train_labels = train_data['label'].tolist()
val_texts = val_data['tweet'].tolist()
val_labels = val_data['label'].tolist()

train_dataset = HateSpeechDataset(train_texts, train_labels)
val_dataset = HateSpeechDataset(val_texts, val_labels)

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir_name = f"outputs/{timestamp}_HindiAbusiveMuRIL_Finetuned"
os.makedirs(output_dir_name, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=f"{output_dir_name}/logs",
    load_best_model_at_end=True,
    save_total_limit=1  

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
    model=muril_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Starting fine-tuning of the Hindi-abusive MuRIL model...")
trainer.train()

fine_tuned_model_path = f"{output_dir_name}/fine_tuned_hindi_abusive_muril"
muril_model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)
print(f"Fine-tuned model saved at {fine_tuned_model_path}")

test_texts = test_df['tweet'].tolist()
test_dataset = HateSpeechDataset(test_texts)

print("Generating predictions on the test set...")
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)

submission_df = pd.DataFrame({
    'index': test_df['index'],
    'prediction': pred_labels.tolist()
})

json_file_path = f'{output_dir_name}/submission_{timestamp}.json'
json_records = submission_df.apply(lambda row: {"index": int(row['index']), "prediction": int(row['prediction'])}, axis=1)
with open(json_file_path, 'w') as json_file:
    for record in json_records:
        json_file.write(json.dumps(record) + '\n')

zip_file_path = f'{output_dir_name}_{timestamp}.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(json_file_path, arcname=f'submission_{timestamp}.json')

print(f"Predictions saved to {zip_file_path}")
