import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.callbacks import EarlyStopping

# Cargar datos
df = pd.read_csv("diario_emociones.csv")

# Balancear clases
dfs = []
for emocion in df["emocion"].unique():
    subset = df[df["emocion"] == emocion]
    dfs.append(resample(subset, replace=True, n_samples=500, random_state=42))
df = pd.concat(dfs, ignore_index=True)

# Codificación de emociones
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emocion"])

# Modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_encoder.classes_))

# Tokenización y preparación de datos
texts = df["texto"].tolist()
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
labels = torch.tensor(y)
dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Entrenamiento
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

for epoch in range(5):  # Ajusta el número de épocas
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Guardar modelo
torch.save(model.state_dict(), 'emociones_model.pth')