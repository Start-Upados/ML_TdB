"""
feature_engineering.py
======================
Engenharia de features, vocabulário e normalização para o classificador
de prioridade da ONG Turma do Bem.

Aplicado server-side em cada request — o cliente só precisa mandar os 9
campos brutos do schema. Este módulo calcula as 12 features derivadas
que o modelo espera (totalizando 21 colunas).

IMPORTANTE: as fórmulas abaixo precisam ser IDÊNTICAS às usadas no notebook
de treino. Se o notebook usou valores ou conjunções diferentes, ajustar aqui.
"""
import pandas as pd


# ============================================================================
# Vocabulário — valores válidos de cada campo categórico
# ============================================================================
VOCAB = {
    'programa': ['apolonas_do_bem', 'dentista_do_bem'],
    'sexo': ['feminino', 'masculino'],
    'tipo_pedido': ['consulta', 'emergencia'],
    'tipo_violencia': ['nenhuma', 'leve', 'grave'],
    'vulnerabilidade': ['baixa', 'media', 'alta'],
    'dano_dentario': ['nenhum', 'leve', 'moderado', 'grave'],
    'tipo_tratamento': ['canal', 'extracao', 'restauracao', 'limpeza'],
}

# Campos obrigatórios no payload de /predict
REQUIRED_FIELDS = [
    'programa', 'tempo_espera', 'sexo', 'idade', 'tipo_violencia',
    'vulnerabilidade', 'dano_dentario', 'tipo_pedido', 'tipo_tratamento',
]

# Aliases aceitos — normalizados antes da validação de vocabulário
ALIASES = {
    'sexo': {
        'f': 'feminino', 'F': 'feminino', 'fem': 'feminino', 'feminina': 'feminino',
        'm': 'masculino', 'M': 'masculino', 'masc': 'masculino',
    },
    'programa': {
        'apolonias_do_bem': 'apolonas_do_bem',
        'apolonia_do_bem': 'apolonas_do_bem',
        'Apolonias do Bem': 'apolonas_do_bem',
        'apoio_social': 'apolonas_do_bem',  # nome interno antigo do frontend
        'Apoio Social': 'apolonas_do_bem',
        'dentista': 'dentista_do_bem',
        'Dentista do Bem': 'dentista_do_bem',
    },
    'tipo_pedido': {
        'urgente': 'emergencia',
        'urgencia': 'emergencia',
        'Urgente': 'emergencia',
    },
}

# ============================================================================
# Mapas ordinais — codificam ordem natural de severidade
# ============================================================================
MAPA_DANO = {'nenhum': 0, 'leve': 1, 'moderado': 2, 'grave': 3}
MAPA_VIOLENCIA = {'nenhuma': 0, 'leve': 1, 'grave': 2}
MAPA_VULNERABILIDADE = {'baixa': 0, 'media': 1, 'alta': 2}


def normalize_input(payload: dict) -> dict:
    """Aplica aliases (F → feminino, urgente → emergencia, etc.)."""
    out = dict(payload)
    for campo, mapa in ALIASES.items():
        if campo in out and out[campo] in mapa:
            out[campo] = mapa[out[campo]]
    return out


def add_business_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame com os 9 campos brutos e devolve com as 12 features
    derivadas adicionadas (total = 21 colunas).

    Features derivadas:
      Scores ordinais:
        - dano_score, violencia_score, vulnerabilidade_score, risco_total
      Flags de idade:
        - idade_50_mais, idade_jovem_alvo
      Conjunções de ALTA — Apolônias do Bem (mulheres):
        - apo_dano_grave_viol_grave
        - apo_dano_grave_idade_50
        - apo_moderado_viol_grave_45
      Conjunções de ALTA — Dentista do Bem (jovens 11-17):
        - dent_vuln_alta_canal_extr
        - dent_vuln_alta_rest_espera_20
      Agregado:
        - regra_alta_acionada
    """
    df = X.copy()

    # ---------- Scores ordinais ----------
    df['dano_score'] = df['dano_dentario'].map(MAPA_DANO).astype(int)
    df['violencia_score'] = df['tipo_violencia'].map(MAPA_VIOLENCIA).astype(int)
    df['vulnerabilidade_score'] = df['vulnerabilidade'].map(MAPA_VULNERABILIDADE).astype(int)
    df['risco_total'] = (
        df['dano_score'] + df['violencia_score'] + df['vulnerabilidade_score']
    )

    # ---------- Flags de idade ----------
    df['idade_50_mais'] = (df['idade'] >= 50).astype(int)
    df['idade_jovem_alvo'] = ((df['idade'] >= 11) & (df['idade'] <= 17)).astype(int)

    # ---------- Apolônias do Bem ----------
    is_apo = (df['programa'] == 'apolonas_do_bem')
    df['apo_dano_grave_viol_grave'] = (
        is_apo
        & (df['dano_dentario'] == 'grave')
        & (df['tipo_violencia'] == 'grave')
    ).astype(int)
    df['apo_dano_grave_idade_50'] = (
        is_apo
        & (df['dano_dentario'] == 'grave')
        & (df['idade'] >= 50)
    ).astype(int)
    df['apo_moderado_viol_grave_45'] = (
        is_apo
        & (df['dano_dentario'] == 'moderado')
        & (df['tipo_violencia'] == 'grave')
        & (df['idade'] >= 45)
    ).astype(int)

    # ---------- Dentista do Bem ----------
    is_dent = (df['programa'] == 'dentista_do_bem')
    df['dent_vuln_alta_canal_extr'] = (
        is_dent
        & (df['vulnerabilidade'] == 'alta')
        & (df['tipo_tratamento'].isin(['canal', 'extracao']))
    ).astype(int)
    df['dent_vuln_alta_rest_espera_20'] = (
        is_dent
        & (df['vulnerabilidade'] == 'alta')
        & (df['tipo_tratamento'] == 'restauracao')
        & (df['tempo_espera'] > 20)
    ).astype(int)

    # ---------- Agregado ----------
    df['regra_alta_acionada'] = df[[
        'apo_dano_grave_viol_grave',
        'apo_dano_grave_idade_50',
        'apo_moderado_viol_grave_45',
        'dent_vuln_alta_canal_extr',
        'dent_vuln_alta_rest_espera_20',
    ]].sum(axis=1)

    return df
