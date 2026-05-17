
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

try:
    model = joblib.load('modelo_ong_prioridade.joblib')
    print("MODELO CARREGADO COM SUCESSO")
except Exception as e:
    print("ERRO AO CARREGAR MODELO:")
    print(e)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online'})

@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return jsonify({
            'prediction': str(prediction)
        })

    except Exception as e:
        return jsonify({
            'erro': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
