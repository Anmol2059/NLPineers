
import kagglehub
kagglehub.login()



nadspoudel_hs_dataset2_path = kagglehub.dataset_download('nadspoudel/hs-dataset2')
nadspoudel_test_dataset_path = kagglehub.dataset_download('nadspoudel/test-dataset')

print('Data source import complete.')

# !pip install -q torch transformers huggingface_hub datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer,TrainingArguments,Trainer
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datasets import load_dataset,DatasetDict
import json
import zipfile

tokenizer=AutoTokenizer.from_pretrained('google/muril-base-cased')

model=AutoModelForSequenceClassification.from_pretrained('google/muril-base-cased')

dataset_file_path='/kaggle/input/hs-dataset2/hs_dataset2.csv'

df = pd.read_csv(dataset_file_path)
df

df['label'].value_counts()

# Preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s\u0900-\u097F]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#apply preprocessing
df['tweet'] = df['tweet'].apply(preprocess_text)

df.columns

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the tweets
train_encodings = tokenizer(train_df['tweet'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_df['tweet'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to tensors
train_labels = torch.tensor(train_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HateSpeechDataset(train_encodings, train_labels)
test_dataset = HateSpeechDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='/kaggle/working/results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='/kaggle/working/logs',
    logging_steps=10,
    save_safetensors=False  

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

import os
os.environ["WANDB_DISABLED"] = "true"

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")


pred_output = trainer.predict(test_dataset)

predictions = pred_output.predictions.argmax(axis=1)  
labels = pred_output.label_ids

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

test_df=pd.read_csv('/kaggle/input/test-dataset/Task-B(indextweet)_label_test.csv')
test_df

test_df['tweet'] = test_df['tweet'].apply(preprocess_text)

test_encodings = tokenizer(
    test_df['tweet'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

class HateSpeechDatasetTestOnly(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

test_dataset = HateSpeechDatasetTestOnly(test_encodings)

predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

submission_df = pd.DataFrame({
    'index': test_df['index'], 
    'prediction': pred_labels.tolist()
})

submission_df = submission_df.sort_values(by='index').reset_index(drop=True)

json_records = submission_df.apply(lambda row: {"index": int(row['index']), "prediction": int(row['prediction'])}, axis=1)

json_file_path = '/kaggle/working/submission.json'
with open(json_file_path, 'w') as json_file:
    for record in json_records:
        json_file.write(json.dumps(record) + '\n')

zip_file_path = '/kaggle/working/res.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(json_file_path, arcname='submission.json')

print("Predictions saved to res.zip")