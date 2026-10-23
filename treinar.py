"""
treinar.py — gera o dataset, treina e salva o modelo v2
======================================================

    python treinar.py                       # só dataset sintético
    python treinar.py --reais casos.csv     # + casos reais rotulados pela equipe

1. Gera N solicitações sintéticas com distribuições parecidas com o público
   real de cada programa (renda, idade, necessidade, sintomas, espera...).
2. Rotula cada uma com o critério do especialista (regras.avaliar).
3. Simula a discordância natural entre avaliadores: perto dos limites de faixa
   (até 6 pontos), parte dos rótulos vai para a faixa vizinha. É isso que faz o
   modelo devolver confiança mais baixa em caso de fronteira, e alta em caso claro.
4. Treina um HistGradientBoosting com probabilidades calibradas.
5. Mede: acurácia contra o critério, matriz de confusão, recall de ALTA e
   distribuição final das prioridades.

CSV de casos reais (opcional): mesmas colunas do /predict v2 + coluna
"prioridade" (ALTA/MEDIA/BAIXA) com a decisão final da equipe (ex.: os
"Reclassificar" e overrides da Central). Esses casos pesam 5x no treino.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from regras import Caso, avaliar, montar_caso, LIMIAR_ALTA, LIMIAR_MEDIA
from feature_engineering import FEATURES, caso_para_linha

CLASSES = ['ALTA', 'MEDIA', 'BAIXA']
VERSAO = 'ml-tdb-v2'


def _escolha(rng, opcoes: dict):
    k = list(opcoes)
    p = np.array(list(opcoes.values()), dtype=float)
    return k[rng.choice(len(k), p=p / p.sum())]


def gerar_caso(rng) -> Caso:
    espera = int(min(180, rng.exponential(18)))
    if rng.random() < 0.6:  # Dentista do Bem
        idade = int(rng.integers(11, 18)) if rng.random() < 0.95 else int(rng.choice([9, 10, 18, 19, 20]))
        renda = _escolha(rng, {'ate_1': 33, '1_a_2': 35, '2_a_3': 17, 'acima_3': 8, 'nao_informada': 7})
        nec = _escolha(rng, {'canal': 14, 'extracao': 11, 'restauracao': 30, 'ortodontia': 20, 'limpeza': 15, 'outro': 10})
        sintoma = _escolha(rng, {'nenhum': 50, 'leve': 10, 'moderado': 28, 'grave': 12})
        vitima = _escolha(rng, {'nao_informado': 95, 'sim': 5})
        enc = _escolha(rng, {'nenhum': 72, 'rede_saude_escola': 24, 'rede_protecao': 4})
        return Caso('dentista_do_bem', idade, 'masculino', renda, nec, vitima, sintoma, enc, espera,
                    impacto_social=bool(rng.random() < 0.15),
                    urgencia=bool(sintoma == 'grave' and rng.random() < 0.4))
    # Apolônias do Bem
    idade = int(np.clip(rng.normal(40, 12), 18, 80)) if rng.random() < 0.97 else int(rng.integers(15, 18))
    vitima = _escolha(rng, {'sim': 70, 'nao': 20, 'nao_informado': 10})
    nec = 'violencia' if vitima == 'sim' else _escolha(rng, {'triagem': 70, 'canal': 8, 'extracao': 7, 'restauracao': 10, 'outro': 5})
    sintoma = _escolha(rng, {'nenhum': 42, 'leve': 8, 'moderado': 32, 'grave': 18})
    renda = _escolha(rng, {'nao_informada': 85, 'ate_1': 7, '1_a_2': 5, '2_a_3': 2, 'acima_3': 1})
    enc = (_escolha(rng, {'nenhum': 60, 'rede_protecao': 30, 'rede_saude_escola': 10}) if vitima != 'nao'
           else _escolha(rng, {'nenhum': 85, 'rede_saude_escola': 15}))
    return Caso('apolonias_do_bem', idade, 'feminino', renda, nec, vitima, sintoma, enc, espera,
                impacto_social=bool(rng.random() < 0.25),
                urgencia=bool(sintoma == 'grave' and rng.random() < 0.4))


def rotular(c: Caso, rng, ruido: float) -> str:
    r = avaliar(c)
    if r.regra_critica or ruido <= 0:
        return r.prioridade
    p = r.pontos
    if r.prioridade == 'ALTA' and p - LIMIAR_ALTA < 6 and rng.random() < ruido:
        return 'MEDIA'
    if r.prioridade == 'MEDIA':
        if LIMIAR_ALTA - p <= 6 and rng.random() < ruido:
            return 'ALTA'
        if p - LIMIAR_MEDIA < 6 and rng.random() < ruido:
            return 'BAIXA'
    if r.prioridade == 'BAIXA' and LIMIAR_MEDIA - p <= 6 and rng.random() < ruido:
        return 'MEDIA'
    return r.prioridade


def carregar_reais(caminho: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(caminho)
    linhas, y = [], []
    for _, row in df.iterrows():
        d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        alvo = str(d.pop('prioridade', '')).upper().replace('É', 'E')
        if alvo not in CLASSES:
            continue
        linhas.append(caso_para_linha(montar_caso(d)))
        y.append(alvo)
    return pd.DataFrame(linhas, columns=FEATURES), pd.Series(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12000)
    ap.add_argument('--ruido', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reais', type=str, default=None)
    ap.add_argument('--saida', type=str, default=str(Path(__file__).parent / 'modelo_ong_prioridade.joblib'))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    casos = [gerar_caso(rng) for _ in range(args.n)]
    X = pd.DataFrame([caso_para_linha(c) for c in casos], columns=FEATURES)
    y_ruido = pd.Series([rotular(c, rng, args.ruido) for c in casos])
    y_criterio = pd.Series([avaliar(c).prioridade for c in casos])
    peso = np.ones(len(X))

    if args.reais:
        Xr, yr = carregar_reais(args.reais)
        print(f'Casos reais carregados: {len(Xr)}')
        X = pd.concat([X, Xr], ignore_index=True)
        y_ruido = pd.concat([y_ruido, yr], ignore_index=True)
        y_criterio = pd.concat([y_criterio, yr], ignore_index=True)
        peso = np.concatenate([peso, np.full(len(Xr), 5.0)])

    idx = np.arange(len(X))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y_ruido)

    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=args.seed,
    )
    modelo = CalibratedClassifierCV(base, method='isotonic', cv=5)
    modelo.fit(X.iloc[i_tr], y_ruido.iloc[i_tr], sample_weight=peso[i_tr])

    pred = modelo.predict(X.iloc[i_te])
    proba = modelo.predict_proba(X.iloc[i_te])
    conf = proba.max(axis=1)

    acc_criterio = accuracy_score(y_criterio.iloc[i_te], pred)
    acc_rotulo = accuracy_score(y_ruido.iloc[i_te], pred)
    dist = pd.Series(pred).value_counts(normalize=True).reindex(CLASSES).fillna(0)

    print('\n=== Contra o critério do especialista (sem ruído) ===')
    print(f'Acurácia: {acc_criterio:.3f}')
    print(classification_report(y_criterio.iloc[i_te], pred, labels=CLASSES, digits=3))
    print('Matriz de confusão (linhas = critério, colunas = modelo):', CLASSES)
    print(confusion_matrix(y_criterio.iloc[i_te], pred, labels=CLASSES))
    print(f'\nAcurácia contra o rótulo com ruído (teto teórico < 1): {acc_rotulo:.3f}')
    print('\nDistribuição das prioridades previstas:')
    print((dist * 100).round(1).to_string())
    print('\nConfiança (prob. da classe prevista): '
          f'p10={np.percentile(conf,10):.2f} mediana={np.median(conf):.2f} p90={np.percentile(conf,90):.2f}')

    metricas = {
        'versao': VERSAO,
        'treinado_em': date.today().isoformat(),
        'n_treino': int(len(i_tr)),
        'n_teste': int(len(i_te)),
        'acuracia_vs_criterio': round(float(acc_criterio), 4),
        'acuracia_vs_rotulo': round(float(acc_rotulo), 4),
        'distribuicao_prevista': {k: round(float(v), 4) for k, v in dist.items()},
        'casos_reais': int((peso > 1).sum()),
    }
    joblib.dump({'modelo': modelo, 'features': FEATURES, 'classes': list(modelo.classes_), 'metricas': metricas},
                args.saida, compress=3)
    Path(args.saida).with_suffix('.metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f'\nModelo salvo em {args.saida}')


if __name__ == '__main__':
    main()
