# Chatbot de Detección de Emociones con BERT

Un chatbot inteligente que detecta emociones en texto usando modelos de lenguaje BERT y ofrece recursos personalizados.

## Características principales
- 🧠 Modelo BERT multilingüe para clasificación de emociones
- 🤖 Sistema de recomendaciones contextuales según la emoción detectada
- 🚀 API REST con FastAPI para integraciones
- 📊 Métricas de evaluación: Accuracy y F1-score
- ⚡ Optimizado para entrenamiento y inferencia

## Requisitos del sistema
- Python 3.8+
- pip
- 8GB+ RAM (16GB recomendado)
- 4GB+ VRAM (para entrenamiento con GPU)

## Instalación
1. Clona el repositorio:
```bash
git clone https://github.com/Zatt010/NeuroHealth
cd NeuroHealth/predictive_model
```
2.Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Entrenamiento del modelo
```bash
python emociones_modelo_trainer.py
```
El modelo entrenado se guardará en:

    /modelo_emociones_final (modelo y tokenizer)

    label_encoder.pkl (codificador de etiquetas)


## Ejecutar la API
```bash
uvicorn api_chatbot:app --reload
```
Accede a la documentación interactiva en:

    http://localhost:8000/docs

    http://localhost:8000/redoc
## Estructura del proyecto
```bash
.
├── api_chatbot.py         # API FastAPI
├── chatbot.py             # Chatbot interactivo
├── emociones_modelo_trainer.py  # Script de entrenamiento
├── modelo_emociones_final/      # Modelo entrenado
├── label_encoder.pkl      # Codificador de etiquetas
├── requirements.txt       # Dependencias
└── diario_emociones.csv   # Dataset de entrenamiento
```
## Uso del modelo
### Desde Python:
```python
from chatbot import predecir_emocion, chatbot_respuesta

texto = "Estoy muy contento con los resultados obtenidos"
emocion, confianza = predecir_emocion(texto)
print(chatbot_respuesta(texto))
```
### Desde la API:
```bash
curl -X POST "http://localhost:8000/emocion" \
-H "Content-Type: application/json" \
-d '{"texto":"Me siento completamente abrumado con todo el trabajo"}'
```
## Respuesta de Ejemplo
```json
{
  "emocion": "estrés",
  "confianza": 0.92
}
```