"""
feature_engineering.py — v2
===========================
Transforma o caso canônico (regras.Caso) nas colunas numéricas do modelo.

O modelo NÃO recebe a pontuação final das regras (senão ele só aprenderia um
"if pontos >= 60"). Ele recebe os fatores crus + o sinal de regra crítica, e
aprende sozinho como combiná-los a partir do dataset rotulado.

Também converte o payload antigo (v1, 9 campos) para o formato novo, para o
front antigo continuar funcionando enquanto o novo não sobe.
"""
from __future__ import annotations

import pandas as pd

from regras import (
    Caso, avaliar, montar_caso,
    NECESSIDADES, ENCAMINHAMENTOS, IDADE_MIN_JOVEM, IDADE_MAX_JOVEM,
)

ORD_RENDA = {'acima_3': 0, '2_a_3': 1, '1_a_2': 2, 'ate_1': 3, 'nao_informada': 1.5}
ORD_VITIMA = {'nao': 0, 'nao_informado': 1, 'sim': 2}
ORD_SINTOMA = {'nenhum': 0, 'leve': 1, 'moderado': 2, 'grave': 3}

FEATURES = (
    ['apolonias', 'idade', 'renda_ord', 'renda_nao_informada', 'vitima_ord', 'sintoma_ord',
     'tempo_espera', 'impacto_social', 'urgencia', 'fora_faixa_programa', 'regra_critica']
    + [f'nec_{n}' for n in NECESSIDADES]
    + [f'enc_{e}' for e in ENCAMINHAMENTOS]
)


def caso_para_linha(c: Caso, regra_critica: bool | None = None) -> dict:
    if regra_critica is None:
        regra_critica = avaliar(c).regra_critica is not None
    apo = c.programa == 'apolonias_do_bem'
    fora = (not apo and not (IDADE_MIN_JOVEM <= c.idade <= IDADE_MAX_JOVEM)) or (apo and c.vitima == 'nao')
    linha = {
        'apolonias': int(apo),
        'idade': c.idade,
        'renda_ord': ORD_RENDA[c.renda],
        'renda_nao_informada': int(c.renda == 'nao_informada'),
        'vitima_ord': ORD_VITIMA[c.vitima],
        'sintoma_ord': ORD_SINTOMA[c.sintoma],
        'tempo_espera': min(c.tempo_espera, 365),
        'impacto_social': int(c.impacto_social),
        'urgencia': int(c.urgencia),
        'fora_faixa_programa': int(fora),
        'regra_critica': int(regra_critica),
    }
    for n in NECESSIDADES:
        linha[f'nec_{n}'] = int(c.necessidade == n)
    for e in ENCAMINHAMENTOS:
        linha[f'enc_{e}'] = int(c.encaminhamento == e)
    return linha


def casos_para_df(casos: list[Caso]) -> pd.DataFrame:
    return pd.DataFrame([caso_para_linha(c) for c in casos], columns=FEATURES)


# ============================================================================
# Compatibilidade com o payload v1 (front antigo)
# ============================================================================
CAMPOS_V1 = ['programa', 'tempo_espera', 'sexo', 'idade', 'tipo_violencia',
             'vulnerabilidade', 'dano_dentario', 'tipo_pedido', 'tipo_tratamento']


def eh_payload_v1(p: dict) -> bool:
    return 'vulnerabilidade' in p and 'renda_familiar' not in p


def v1_para_v2(p: dict) -> dict:
    renda = {'alta': 'ate_1', 'media': '1_a_2', 'baixa': '2_a_3'}.get(str(p.get('vulnerabilidade', '')).lower(), 'nao_informada')
    vit = {'grave': 'sim', 'leve': 'sim'}.get(str(p.get('tipo_violencia', '')).lower(), 'nao_informado')
    sintoma = {'grave': 'grave', 'moderado': 'moderado', 'leve': 'leve'}.get(str(p.get('dano_dentario', '')).lower(), 'nenhum')
    if str(p.get('tipo_pedido', '')).lower() in ('emergencia', 'urgente') and sintoma in ('nenhum', 'leve'):
        sintoma = 'moderado'
    return {
        'programa': p.get('programa'),
        'idade': p.get('idade'),
        'sexo': p.get('sexo'),
        'renda_familiar': renda,
        'necessidade': p.get('tipo_tratamento'),
        'vitima_violencia': vit,
        'sintoma': sintoma,
        'tempo_espera': p.get('tempo_espera', 0),
        'texto': '',
    }


__all__ = ['FEATURES', 'caso_para_linha', 'casos_para_df', 'eh_payload_v1', 'v1_para_v2', 'montar_caso', 'CAMPOS_V1']
