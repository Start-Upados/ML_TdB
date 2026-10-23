"""
app.py — API REST do classificador de prioridade da Turma do Bem (v2)

Endpoints:
    GET  /health   — status do serviço e métricas do modelo
    GET  /schema   — contrato de entrada + exemplos
    POST /predict  — classifica uma solicitação (aceita payload v2 e o antigo v1)

Decisão final:
    1. Regra crítica acionada (regras.py)  -> ALTA
    2. Caso contrário                      -> classe prevista pelo modelo
    3. Fora do perfil do programa          -> nunca ALTA (salvo urgência clínica)
"""
import os
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from regras import (
    montar_caso, avaliar, RENDAS, NECESSIDADES, VITIMA, SINTOMAS,
    LIMIAR_ALTA, LIMIAR_MEDIA,
)
from feature_engineering import FEATURES, caso_para_linha, eh_payload_v1, v1_para_v2, CAMPOS_V1

MODEL_PATH = os.environ.get('MODEL_PATH', str(Path(__file__).parent / 'modelo_ong_prioridade.joblib'))
VERSAO = 'ml-tdb-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

_default_origins = 'http://localhost:5173,http://localhost:3000'
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('CORS_ALLOWED_ORIGINS', _default_origins).split(',') if o.strip()]
CORS(app, resources={r'/*': {'origins': ALLOWED_ORIGINS}}, methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'])
logger.info(f'CORS habilitado para origens: {ALLOWED_ORIGINS}')

# ============================================================================
# Modelo
# ============================================================================
try:
    _pacote = joblib.load(MODEL_PATH)
    model = _pacote['modelo']
    MODEL_FEATURES = _pacote['features']
    MODEL_CLASSES = [str(c) for c in _pacote['classes']]
    METRICAS = _pacote.get('metricas', {})
    if MODEL_FEATURES != FEATURES:
        raise RuntimeError('As colunas do modelo não batem com feature_engineering.FEATURES — rode treinar.py de novo')
    MODEL_LOADED = True
    logger.info(f'Modelo {METRICAS.get("versao")} carregado. Classes: {MODEL_CLASSES}')
except Exception as e:  # noqa: BLE001
    logger.error(f'Falha ao carregar modelo: {e}')
    model, MODEL_CLASSES, METRICAS, MODEL_LOADED = None, [], {}, False


# ============================================================================
# Erros sempre em JSON
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
    return jsonify({'error': 'Erro interno do servidor', 'type': type(e).__name__}), 500


# ============================================================================
# Endpoints
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    if MODEL_LOADED:
        return jsonify({'status': 'ok', 'model_loaded': True, 'versao': VERSAO,
                        'classes': MODEL_CLASSES, 'metricas': METRICAS}), 200
    return jsonify({'status': 'error', 'model_loaded': False}), 503


EXEMPLO_JOVEM = {
    'programa': 'Dentista do Bem', 'idade': 16, 'sexo': 'masculino',
    'renda_familiar': 'Ate 1 salario minimo', 'necessidade': 'Tratamento de canal',
    'texto': 'Sente dor quando mastiga faz duas semanas', 'tempo_espera': 12, 'como_soube': 'Escola',
}
EXEMPLO_MULHER = {
    'programa': 'Apolônias do Bem', 'idade': 42, 'sexo': 'feminino',
    'necessidade': 'Vitima de violencia com denticao afetada',
    'texto': 'Perdi dois dentes depois de uma agressão', 'tempo_espera': 3, 'como_soube': 'Delegacia/CREAS',
}


@app.route('/schema', methods=['GET'])
def schema():
    return jsonify({
        'versao': VERSAO,
        'obrigatorios': ['idade', 'programa ou sexo'],
        'campos': {
            'programa': 'Dentista do Bem | Apolônias do Bem (ou dentista_do_bem / apolonias_do_bem)',
            'idade': 'inteiro, anos',
            'sexo': 'feminino | masculino',
            'renda_familiar': f'texto do formulário ("Ate 1 salario minimo"...) ou {RENDAS}',
            'necessidade': f'texto do formulário ("Tratamento de canal"...) ou {NECESSIDADES}',
            'vitima_violencia': f'{VITIMA} (opcional; o formulário das Apolônias já manda na necessidade)',
            'texto': 'descrição/observações livres do paciente — sintomas, violência e impacto são lidos daqui',
            'sintoma': f'{SINTOMAS} (opcional; o texto também é lido)',
            'tempo_espera': 'inteiro, dias desde a solicitação',
            'como_soube': 'texto do formulário ("Delegacia/CREAS", "Posto de saude", "Escola"...)',
        },
        'faixas': {'ALTA': f'>= {LIMIAR_ALTA} pontos ou regra crítica',
                   'MEDIA': f'{LIMIAR_MEDIA}-{LIMIAR_ALTA - 1}', 'BAIXA': f'< {LIMIAR_MEDIA}'},
        'payload_v1_aceito': CAMPOS_V1,
        'example_request': EXEMPLO_JOVEM,
        'exemplo_apolonias': EXEMPLO_MULHER,
    }), 200


def _validar(payload: dict):
    if payload.get('idade') in (None, ''):
        return 'Campo obrigatório faltando: idade'
    try:
        idade = int(payload['idade'])
    except (TypeError, ValueError):
        return 'idade deve ser um inteiro'
    if not 0 < idade < 120:
        return 'idade fora do intervalo esperado'
    try:
        if int(payload.get('tempo_espera') or 0) < 0:
            return 'tempo_espera não pode ser negativo'
    except (TypeError, ValueError):
        return 'tempo_espera deve ser um inteiro (dias)'
    if not payload.get('programa') and not payload.get('sexo'):
        return 'Informe programa ou sexo'
    return None


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não disponível'}), 503

    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'Payload deve ser um objeto JSON'}), 400

    formato = 'v1' if eh_payload_v1(payload) else 'v2'
    if formato == 'v1':
        payload = v1_para_v2(payload)

    erro = _validar(payload)
    if erro:
        return jsonify({'error': erro}), 400

    caso = montar_caso(payload)
    criterio = avaliar(caso)

    df = pd.DataFrame([caso_para_linha(caso, regra_critica=criterio.regra_critica is not None)], columns=FEATURES)
    proba = model.predict_proba(df)[0]
    probabilidades = {c: round(float(p), 4) for c, p in zip(MODEL_CLASSES, proba)}
    pred_modelo = max(probabilidades, key=probabilidades.get)

    if criterio.regra_critica:
        prediction, origem = 'ALTA', 'regra_critica'
    else:
        prediction, origem = pred_modelo, 'modelo'
        fora = any('elegibilidade' in a for a in criterio.alertas)
        if fora and prediction == 'ALTA' and not caso.urgencia:
            prediction, origem = 'MEDIA', 'modelo_limitado_por_elegibilidade'

    entrada = asdict(caso)
    entrada.pop('termo_sintoma', None)

    return jsonify({
        'prediction': prediction,
        'probabilities': {k: probabilidades.get(k, 0.0) for k in ('ALTA', 'MEDIA', 'BAIXA')},
        'confianca': probabilidades.get(prediction, 0.0),
        'origem': origem,
        'pontuacao': criterio.pontos,
        'prioridade_criterio': criterio.prioridade,
        'regra_critica': criterio.regra_critica,
        'fatores': [{'fator': f, 'pontos': p} for f, p in criterio.fatores],
        'alertas': criterio.alertas,
        'entrada_interpretada': entrada,
        'formato_entrada': formato,
        'modelo_versao': VERSAO,
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
