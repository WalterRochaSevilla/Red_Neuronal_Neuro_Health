from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pickle
import os
import requests
import zipfile

# ========== CONFIGURACIÓN ==========
MODEL_DIR = "modelo_emociones_final"
MODEL_ZIP = "modelo_emociones_final.zip"
DRIVE_FILE_ID = "1IzeB1UxlVM_KrN_3QUxR5toPAlqeAU-0"  # ID del archivo en Google Drive
MODEL_ZIP_PATH = os.path.join(".", MODEL_ZIP)

# ========== UTILIDADES ==========
def descargar_descomprimir_modelo():
    if not os.path.exists(MODEL_DIR):
        print("🔽 Descargando modelo desde Google Drive...")
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        response = session.get(URL, params={'id': DRIVE_FILE_ID}, stream=True)
        token = get_confirm_token(response)

        if token:
            response = session.get(URL, params={'id': DRIVE_FILE_ID, 'confirm': token}, stream=True)

        guardar_como_zip(response, MODEL_ZIP_PATH)
        descomprimir_zip(MODEL_ZIP_PATH, ".")
        os.remove(MODEL_ZIP_PATH)
        print("✅ Modelo descargado y descomprimido.")
    else:
        print("✅ Modelo ya disponible localmente.")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def guardar_como_zip(response, destino):
    CHUNK_SIZE = 32768
    with open(destino, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def descomprimir_zip(path_zip, path_destino):
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_destino)

# ========== DESCARGAR MODELO ==========
descargar_descomprimir_modelo()

# ========== CARGAR MODELO ==========
print("📦 Cargando modelo...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, from_safetensors=True)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model.eval()
print("✅ Modelo cargado y listo.")

# ========== API FASTAPI ==========
app = FastAPI()

class EntradaTexto(BaseModel):
    texto: str

@app.post("/emocion")
def detectar_emocion(datos: EntradaTexto):
    inputs = tokenizer(datos.texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
    idx = np.argmax(probs)
    emocion = label_encoder.inverse_transform([idx])[0]
    confianza = float(probs[idx])
    return {"emocion": emocion, "confianza": confianza}
