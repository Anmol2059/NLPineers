
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import zipfile
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_model = AutoModel.from_pretrained("xlm-roberta-base").to("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('/codes/CHIPSAL/dataset/train.csv')
test_df = pd.read_csv('/codes/CHIPSAL/dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str)
test_df['tweet'] = test_df['tweet'].astype(str)

def get_xlm_roberta_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(xlmr_model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model
    with torch.no_grad():
        outputs = xlmr_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Mean pooling of embeddings
    return embeddings

print("Extracting embeddings from XLM-Roberta model...")
train_embeddings = np.array([get_xlm_roberta_embeddings(text) for text in tqdm(train_df['tweet'], desc="Train embeddings")])
test_embeddings = np.array([get_xlm_roberta_embeddings(text) for text in tqdm(test_df['tweet'], desc="Test embeddings")])

train_labels = train_df['label'].values  

scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

print("Training Logistic Regression classifier on embeddings...")
log_reg_clf = LogisticRegression(max_iter=1000)
log_reg_clf.fit(train_embeddings, train_labels)

pred_labels = log_reg_clf.predict(test_embeddings)

submission_df = pd.DataFrame({
    'index': test_df['index'],  
    'prediction': pred_labels.tolist()
})

submission_df = submission_df.sort_values(by='index').reset_index(drop=True)

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir_name = f"{timestamp}_XLMRoberta_LogReg"

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
