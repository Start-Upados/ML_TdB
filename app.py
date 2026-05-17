
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ===============================
# CARREGAR MODELO
# ===============================

model = joblib.load('modelo_ong_prioridade.joblib')

# ===============================
# HEALTH CHECK
# ===============================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online'
    })

# ===============================
# PREDICTION
# ===============================

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    mapa_classes = {
    0: "BAIXA",
    1: "MEDIA",
    2: "ALTA"
    }

    return jsonify({
    'prediction': mapa_classes.get(int(prediction), str(prediction))
    })

# ===============================
# START SERVER
# ===============================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
