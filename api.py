# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_respuesta

app = Flask(__name__)
CORS(app)  # Habilita CORS para todos los dominios

@app.route('/analizar', methods=['POST'])
def analizar_texto():
    try:
        data = request.get_json()
        texto = data.get('texto', '')
        
        if not texto:
            return jsonify({'error': 'Texto vacío'}), 400
            
        respuesta = chatbot_respuesta(texto)
        
        return jsonify({
            'texto_usuario': texto,
            'respuesta_chatbot': respuesta.split('\n'),
            'estado': 'exito'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'estado': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)