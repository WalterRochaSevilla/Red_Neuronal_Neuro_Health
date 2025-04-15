import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Cargar modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=6)  # Ajusta el número de etiquetas
model.load_state_dict(torch.load('emociones_model.pth'))
model.eval()

# Codificación de emociones (mismo orden que en el entrenamiento)
label_encoder = LabelEncoder()
label_encoder.fit(['enojo', 'estrés', 'felicidad', 'indefinido', 'tristeza', 'ansiedad'])

def predecir_emocion(texto, umbral_confianza=0.5):
    inputs = tokenizer(texto, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
    emocion_id = np.argmax(probs)
    confianza = probs[emocion_id]

    if confianza < umbral_confianza:
        return "indefinido", confianza
    return label_encoder.inverse_transform([emocion_id])[0], confianza

def chatbot_respuesta(texto):
    emocion, confianza = predecir_emocion(texto)

    recursos = {
        "tristeza": {
            "mensaje": " Pareces sentir tristeza. Te recomiendo:",
            "acciones": ["Meditación guiada 'Alivio emocional' (10 min)", "Ejercicio: Escribe una carta para liberar emociones"]
        },
        "ansiedad": {
            "mensaje": "🟠 Detecto señales de ansiedad. Prueba:",
            "acciones": ["Técnica de respiración 4-7-8 (instrucciones)", "Ejercicio de grounding: 5-4-3-2-1"]
        },
        "estrés": {
            "mensaje": "🟢 Sugerencias para manejar el estrés:",
            "acciones": ["Meditación 'Libera tensiones' (15 min)", "Prioriza tareas con matriz Eisenhower"]
        },
        "enojo": {
            "mensaje": " Detecto frustración. Intenta:",
            "acciones": ["Ejercicio físico de alta intensidad", "Técnica de pausa consciente de 5 minutos"]
        },
        "felicidad": {
            "mensaje": "🟡 ¡Me alegra verte así! Mantén esto:",
            "acciones": ["Registra este momento en tu diario positivo", "Comparte tu estado con alguien especial"]
        },
        "indefinido": {
            "mensaje": "⚪️ Necesito entenderte mejor. ¿Podrías:",
            "acciones": ["Describir cómo te sientes con más detalle?", "Contarme qué ha pasado recientemente?"]
        }
    }

    respuesta = recursos.get(emocion, recursos["indefinido"])
    return (
        f"{respuesta['mensaje']}\n" +
        "\n".join([f"- {accion}" for accion in respuesta['acciones']]) +
        f"\n\nConfianza del modelo: {confianza:.2%}"
    )

test_cases = [
    "No tengo ganas de salir de la cama hoy",
    "Estoy harto de las reuniones inútiles",
    "Me palpita el corazón muy rápido",
    "Logré terminar todos mis pendientes a tiempo!",
    "El tráfico me hizo llegar tarde otra vez"
]

for caso in test_cases:
    print(f" Usuario: {caso}")
    print(f" Chatbot:\n{chatbot_respuesta(caso)}\n")
    print("―" * 50)