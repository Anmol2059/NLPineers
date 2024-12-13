
from transformers import AutoTokenizer, AutoModel
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import zipfile
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased").to("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('/codes/CHIPSAL/dataset/train.csv')
test_df = pd.read_csv('/codes/CHIPSAL/dataset/test.csv')

train_df['tweet'] = train_df['tweet'].astype(str)
test_df['tweet'] = test_df['tweet'].astype(str)

def get_muril_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(muril_model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model
    with torch.no_grad():
        outputs = muril_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Mean pooling of embeddings
    return embeddings

print("Extracting embeddings from MuRIL model...")
train_embeddings = np.array([get_muril_embeddings(text) for text in tqdm(train_df['tweet'], desc="Train embeddings")])
test_embeddings = np.array([get_muril_embeddings(text) for text in tqdm(test_df['tweet'], desc="Test embeddings")])

train_labels = train_df['label'].values

scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

print("Training TabNet classifier on embeddings without PCA...")
tabnet_clf = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-3), verbose=1)

tabnet_clf.fit(
    train_embeddings,
    train_labels,
    max_epochs=150,      
    patience=10,        
    batch_size=256,     
    drop_last=False
)

pred_labels = tabnet_clf.predict(test_embeddings)

pred_labels = np.where(pred_labels == 1, 1, 0)

submission_df = pd.DataFrame({
    'index': test_df['index'], 
    'prediction': pred_labels.tolist()
})

submission_df = submission_df.sort_values(by='index').reset_index(drop=True)

timestamp = datetime.now().strftime("%d_%b_%H_%M_%S")
output_dir_name = f"{timestamp}_Muriltabnet_nofinetune"

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
