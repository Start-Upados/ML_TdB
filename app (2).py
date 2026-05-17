
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('modelo_ong_prioridade.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online'})

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return jsonify({
        'prediction': str(prediction)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
