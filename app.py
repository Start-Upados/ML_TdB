"""
app.py — API REST para o classificador de prioridade da ONG Turma do Bem.

Endpoints:
    GET  /health   — status do serviço
    GET  /schema   — valores válidos de cada campo + exemplo
    POST /predict  — recebe JSON com 9 campos brutos, retorna prioridade

Como rodar:
    python app.py

Requisitos no mesmo diretório:
    - model_pipeline.joblib
    - feature_engineering.py
"""
import os
import logging
import traceback
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

from feature_engineering import (
    add_business_features,
    normalize_input,
    REQUIRED_FIELDS,
    VOCAB,
    ALIASES,
)

# ============================================================================
# Configuração
# ============================================================================
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    str(Path(__file__).parent / 'modelo_ong_prioridade.joblib')
)

# Mapa de classes — DEVE bater com le_y.classes_ do notebook de treino.
# LabelEncoder do sklearn ordena alfabeticamente: ['ALTA', 'BAIXA', 'MEDIA']
# ou seja, 0→ALTA, 1→BAIXA, 2→MEDIA.
# Se o seu notebook usou outra ordem, ajuste aqui.
MAPA_CLASSES = {0: 'ALTA', 1: 'BAIXA', 2: 'MEDIA'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# Carrega o modelo no startup
# ============================================================================
try:
    logger.info(f'Carregando modelo de {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)
    CLASSES = list(getattr(model, 'classes_', []))
    MODEL_LOADED = True
    logger.info(f'Modelo carregado. Classes: {CLASSES}')
except Exception as e:
    logger.error(f'Falha ao carregar modelo: {e}')
    model = None
    CLASSES = []
    MODEL_LOADED = False


# ============================================================================
# Handlers globais — sempre retornam JSON, nunca HTML
# ============================================================================
@app.errorhandler(404)
def not_found(_e):
    return jsonify({'error': 'Endpoint não encontrado'}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({'error': 'Método HTTP não permitido'}), 405


@app.errorhandler(Exception)
def handle_unexpected(e):
    logger.error(f'Erro não tratado: {e}\n{traceback.format_exc()}')
    return jsonify({
        'error': 'Erro interno do servidor',
        'type': type(e).__name__,
        'detail': str(e),
    }), 500


# ============================================================================
# Endpoints
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    if MODEL_LOADED:
        return jsonify({
            'status': 'ok',
            'model_loaded': True,
            'classes': CLASSES,
        }), 200
    return jsonify({'status': 'error', 'model_loaded': False}), 503


@app.route('/schema', methods=['GET'])
def schema():
    """Documenta o contrato da API — útil pro frontend e pra debug."""
    return jsonify({
        'required_fields': REQUIRED_FIELDS,
        'vocabulary': VOCAB,
        'aliases_accepted': ALIASES,
        'example_request': {
            'programa': 'apolonas_do_bem',
            'tempo_espera': 30,
            'sexo': 'feminino',
            'idade': 70,
            'tipo_violencia': 'grave',
            'vulnerabilidade': 'alta',
            'dano_dentario': 'grave',
            'tipo_pedido': 'emergencia',
            'tipo_tratamento': 'canal',
        },
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não disponível'}), 503

    # 1. Parse JSON
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({'error': 'JSON inválido', 'detail': str(e)}), 400

    if not isinstance(payload, dict):
        return jsonify({'error': 'Payload deve ser um objeto JSON'}), 400

    # 2. Normaliza aliases (F → feminino, urgente → emergencia, etc.)
    payload = normalize_input(payload)

    # 3. Valida campos obrigatórios
    faltando = [f for f in REQUIRED_FIELDS if f not in payload]
    if faltando:
        return jsonify({
            'error': 'Campos obrigatórios faltando',
            'missing': faltando,
            'required_fields': REQUIRED_FIELDS,
        }), 400

    # 4. Valida vocabulário das categóricas
    invalid = {}
    for campo, valores_validos in VOCAB.items():
        if payload[campo] not in valores_validos:
            invalid[campo] = {
                'received': payload[campo],
                'allowed': valores_validos,
            }
    if invalid:
        return jsonify({
            'error': 'Valores inválidos',
            'invalid': invalid,
        }), 400

    # 5. Valida tipos numéricos
    try:
        payload['idade'] = int(payload['idade'])
        payload['tempo_espera'] = int(payload['tempo_espera'])
    except (ValueError, TypeError):
        return jsonify({
            'error': 'idade e tempo_espera devem ser inteiros',
        }), 400

    # 6. Monta DataFrame APENAS com os 9 campos brutos
    df = pd.DataFrame([{f: payload[f] for f in REQUIRED_FIELDS}])

    # 7. Feature engineering server-side (gera as 12 features derivadas)
    try:
        df_full = add_business_features(df)
    except Exception as e:
        logger.exception('Erro na feature engineering')
        return jsonify({
            'error': 'Erro na feature engineering',
            'detail': str(e),
        }), 500

    # 8. Predição
    try:
        pred_int = int(model.predict(df_full)[0])
        pred_label = MAPA_CLASSES.get(pred_int, str(pred_int))
        result = {'prediction': pred_label}

        # Se o modelo expõe probabilidades, devolve junto — útil pra confiança
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df_full)[0]
            result['probabilities'] = {
                MAPA_CLASSES.get(int(cls), str(cls)): round(float(p), 4)
                for cls, p in zip(CLASSES, proba)
            }
        return jsonify(result), 200
    except Exception as e:
        logger.exception('Erro na predição')
        return jsonify({
            'error': 'Erro na predição',
            'detail': str(e),
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f'Servidor iniciando na porta {port}')
    app.run(host='0.0.0.0', port=port, debug=False)
