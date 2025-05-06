import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

# Cargar y balancear datos
df = pd.read_csv("diario_emociones.csv")
dfs = [df[df["emocion"] == e].sample(500, replace=True, random_state=42) for e in df["emocion"].unique()]
df = pd.concat(dfs, ignore_index=True)

# Codificación de etiquetas
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emocion"])
df = df[["texto", "label"]]

# Separar en train/test
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenización
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
def tokenize(example):
    return tokenizer(example["texto"], truncation=True, padding="max_length", max_length=128)
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Modelo
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)

# Entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # <- aquí va el early stopping
)

trainer.train()
trainer.save_model("modelo_emociones_final")
tokenizer.save_pretrained("modelo_emociones_final")

# Guardar label encoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
