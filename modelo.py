import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. Cargar y preparar los datos
df = pd.read_csv("dataset_emociones_chatbot.csv")

# Distribución original
print("Distribución original de emociones:")
print(df['emocion'].value_counts())

# 2. Preprocesamiento de texto
nlp = spacy.load('es_core_news_sm')

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'@\w+|\#\w+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    return texto.strip()

df['texto_limpio'] = df['texto'].apply(limpiar_texto)

# 3. Codificar etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emocion'])

# 4. Tokenización y secuenciación
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['texto_limpio'])
X_seq = tokenizer.texts_to_sequences(df['texto_limpio'])
X_seq = pad_sequences(X_seq, maxlen=100)

# 5. Dividir datos ORIGINALES (antes de balancear)
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Balancear  el conjunto de entrenamiento
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# nuevas distribuciones
print("\nDistribución después de balanceo:")
unique, counts = np.unique(y_train_res, return_counts=True)
for emotion, count in zip(label_encoder.classes_, counts):
    print(f"{emotion}: {count}")

# 7.  el modelo
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=100),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_emociones.h5', monitor='val_accuracy', save_best_only=True)
]

# 9. Entrenamiento
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# 10. Evaluación
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Guardar el label encoder
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nComponentes guardados:")
print("- mejor_modelo_emociones.h5")
print("- tokenizer.pickle")
print("- label_encoder.pickle")

# 11. Sistema de predicción del chatbot 
def predecir_emocion(texto, umbral=0.7):
    try:
        texto_limpio = limpiar_texto(texto)
        seq = tokenizer.texts_to_sequences([texto_limpio])
        if not seq[0]:
            return "indefinido", 0.0
            
        padded = pad_sequences(seq, maxlen=100)
        proba = model.predict(padded, verbose=0)[0]
        
        if np.max(proba) < umbral:
            return "indefinido", np.max(proba)
        return label_encoder.inverse_transform([np.argmax(proba)])[0], np.max(proba)
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return "error", 0.0

def chatbot_respuesta(texto):
    emocion, confianza = predecir_emocion(texto)
    
    recursos = {
        "tristeza": {
            "mensaje": "🔵 Detecto que podrías estar sintiendo tristeza",
            "acciones": [
                "Habla con alguien de confianza sobre cómo te sientes",
                "Prueba escribir un diario de emociones",
                "Escucha música que te levante el ánimo"
            ]
        },
        "ansiedad": {
            "mensaje": "🟠 Detecto señales de ansiedad",
            "acciones": [
                "Practica respiración 4-7-8 (inhala 4s, mantén 7s, exhala 8s)",
                "Ejercicio de grounding: nombra 5 cosas que ves, 4 que tocas...",
                "Haz una lista de cosas que puedes controlar"
            ]
        },
        "felicidad": {
            "mensaje": "🟡 ¡Me alegra verte feliz!",
            "acciones": [
                "Aprovecha para hacer algo que disfrutes",
                "Comparte tu alegría con alguien más",
                "Registra este momento en tu diario positivo"
            ]
        },
        "depresion": {
            "mensaje": "⚫️ Detecto señales de depresión",
            "acciones": [
                "Considera hablar con un profesional de salud mental",
                "Intenta mantener rutinas básicas (sueño, alimentación)",
                "Busca apoyo en personas cercanas"
            ]
        },
        "indefinido": {
            "mensaje": "⚪️ No estoy seguro de cómo te sientes",
            "acciones": [
                "¿Podrías describirlo con otras palabras?",
                "Cuéntame más sobre lo que estás experimentando"
            ]
        },
        "error": {
            "mensaje": "🔴 Hubo un error al procesar tu mensaje",
            "acciones": [
                "Por favor intenta expresarte de otra manera",
                "Reinicia la conversación"
            ]
        }
    }
    
    respuesta = recursos.get(emocion, recursos["indefinido"])
    output = f"{respuesta['mensaje']} (Confianza: {confianza:.1%})\n\n"
    
    if respuesta['acciones']:
        output += "Te sugiero:\n" + "\n".join([f"• {a}" for a in respuesta['acciones']])
    
    return output

# Pruebas mejoradas
test_cases = [
    "Estoy eufórico con los resultados!",
    "No tengo ganas de nada, solo quiero llorar",
    "Me siento atrapado en mis pensamientos",
    "Logré todos mis objetivos del mes!",
    "Todo me da igual últimamente",
    "Palabras desconocidas para el modelo"  # Prueba con texto no visto
]

print("\n" + "="*50)
print("PRUEBAS DEL CHATBOT DE EMOCIONES")
print("="*50 + "\n")

for i, caso in enumerate(test_cases, 1):
    print(f"Prueba #{i}:")
    print(f"🧑 Usuario: {caso}")
    print(f"🤖 Chatbot:\n{chatbot_respuesta(caso)}")
    print("-"*50 + "\n")