"""
regras.py — Regras de prioridade da Turma do Bem (v2)
=====================================================

Um lugar só para:
  1. Normalizar a entrada (aceita os textos crus do formulário do site).
  2. Ler a descrição livre do paciente (sintomas, violência, impacto social).
  3. Calcular a PONTUAÇÃO (0-100) e as REGRAS CRÍTICAS.

A pontuação é o "critério do especialista": é com ela que o dataset de treino é
rotulado (treinar.py) e é ela que explica cada decisão na tela da Central.

Faixas:   ALTA >= 65   |   MÉDIA 40-64   |   BAIXA < 40
Regra crítica acionada  -> ALTA direto (casos que não podem esperar fila).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ============================================================================
# Vocabulário (valores canônicos)
# ============================================================================
PROGRAMAS = ['dentista_do_bem', 'apolonias_do_bem']
RENDAS = ['ate_1', '1_a_2', '2_a_3', 'acima_3', 'nao_informada']
NECESSIDADES = ['canal', 'extracao', 'restauracao', 'ortodontia', 'limpeza', 'outro', 'violencia', 'triagem']
VITIMA = ['sim', 'nao', 'nao_informado']
SINTOMAS = ['nenhum', 'leve', 'moderado', 'grave']
ENCAMINHAMENTOS = ['nenhum', 'rede_saude_escola', 'rede_protecao']

LIMIAR_ALTA = 65
LIMIAR_MEDIA = 40

IDADE_MIN_JOVEM, IDADE_MAX_JOVEM = 11, 17


# ============================================================================
# Helpers de texto
# ============================================================================
def sem_acento(v) -> str:
    s = '' if v is None else str(v)
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', s.lower()).strip()


# ============================================================================
# Normalização dos campos (aceita o texto do formulário e os valores canônicos)
# ============================================================================
def normalizar_programa(v) -> str | None:
    s = sem_acento(v).replace('_', ' ')
    if not s:
        return None
    if 'apolon' in s or 'apoio social' in s or 'mulher' in s:
        return 'apolonias_do_bem'
    if 'dentista' in s or 'jovem' in s:
        return 'dentista_do_bem'
    return None


def normalizar_sexo(v) -> str | None:
    s = sem_acento(v)
    if s in ('f', 'fem') or s.startswith('femin') or s.startswith('mulher'):
        return 'feminino'
    if s in ('m', 'masc') or s.startswith('mascul') or s.startswith('homem'):
        return 'masculino'
    return None


def normalizar_renda(v) -> str:
    s = sem_acento(v).replace('_', ' ')
    if not s or 'nao se aplica' in s or 'nao informad' in s:
        return 'nao_informada'
    if s in RENDAS:
        return s
    if s.startswith('ate 1') or 'ate 1 salario' in s or s == 'ate 1':
        return 'ate_1'
    if s.startswith('1 a 2'):
        return '1_a_2'
    if s.startswith('2 a 3'):
        return '2_a_3'
    if s.startswith('acima') or 'mais de 3' in s:
        return 'acima_3'
    return 'nao_informada'


def normalizar_necessidade(v) -> str:
    s = sem_acento(v)
    if not s:
        return 'outro'
    if s.replace(' ', '_') in NECESSIDADES:
        return s.replace(' ', '_')
    if 'violencia' in s or 'vitima' in s:
        return 'violencia'
    if 'triagem' in s:
        return 'triagem'
    if 'canal' in s or 'endodont' in s:
        return 'canal'
    if 'extra' in s:
        return 'extracao'
    if 'restaura' in s:
        return 'restauracao'
    if 'ortodon' in s or 'aparelho' in s:
        return 'ortodontia'
    if 'limpeza' in s or 'prevenc' in s or 'profilax' in s:
        return 'limpeza'
    return 'outro'


def normalizar_vitima(v, necessidade: str) -> str:
    s = sem_acento(v)
    if s in ('sim', 's', 'true', '1', 'grave', 'leve'):
        return 'sim'
    if s in ('nao', 'n', 'false', '0'):
        return 'nao'
    # Formulário das Apolônias grava a resposta dentro da "necessidade"
    if necessidade == 'violencia':
        return 'sim'
    if necessidade == 'triagem':
        return 'nao'
    return 'nao_informado'


def normalizar_encaminhamento(v) -> str:
    s = sem_acento(v)
    if not s:
        return 'nenhum'
    if s in ENCAMINHAMENTOS:
        return s
    if 'delegacia' in s or 'creas' in s or 'cras' in s or 'conselho tutelar' in s or 'abrigo' in s:
        return 'rede_protecao'
    if 'posto' in s or 'ubs' in s or 'escola' in s or 'saude' in s:
        return 'rede_saude_escola'
    return 'nenhum'


# ============================================================================
# Leitura da descrição livre
# ============================================================================
# Cada termo é procurado com "negação" logo antes ("sem dor", "não tenho dor").
_NEGACAO = re.compile(r'\b(sem|nao|nenhum|nenhuma|nunca)\b(\W+\w+){0,2}\W*$')

# Sinais de urgência: infecção ou trauma — vão direto para ALTA
TERMOS_URGENCIA = [
    r'absces', r'\bpus\b', r'inchad', r'inchac', r'febre', r'sangr', r'infecc', r'infeccion',
    r'rosto inchado', r'nao consigo (comer|dormir|abrir a boca)', r'dor insuportavel',
]
TERMOS_GRAVE = TERMOS_URGENCIA + [
    r'quebr', r'fratur', r'arrancad', r'perdi (um |o |os |meu |meus )?dente', r'dente caiu', r'dentes cairam',
    r'dor (muito )?forte', r'muita dor', r'dor insuportavel', r'dor intensa', r'nao consigo (comer|dormir|mastigar|falar|abrir a boca)',
]
TERMOS_MODERADO = [
    r'\bdor\b', r'\bdoi\b', r'doendo', r'sensibilidade', r'\bcarie', r'buraco', r'inflama', r'dente mole',
    r'dente podre', r'gengiva', r'dificuldade (para|de|pra) (comer|mastigar|falar)',
]
TERMOS_LEVE = [r'mancha', r'tartaro', r'mau halito', r'amarelad', r'torto']
TERMOS_VIOLENCIA = [
    r'violencia', r'agress', r'agredid', r'apanhe', r'\bsoco', r'espancad', r'medida protetiva',
    r'(marido|companheiro|ex) (me )?(bateu|agrediu)', r'lei maria da penha', r'vitima',
]
TERMOS_IMPACTO = [
    r'vergonha', r'nao consigo sorrir', r'nao sorrio', r'sorrir', r'emprego', r'entrevista', r'trabalho',
    r'bullying', r'zoam', r'autoestima', r'auto estima', r'isolad',
]


def _encontrou(texto: str, padroes: list[str]) -> str | None:
    for p in padroes:
        for m in re.finditer(p, texto):
            antes = texto[max(0, m.start() - 25):m.start()]
            if _NEGACAO.search(antes):
                continue
            return m.group(0)
    return None


@dataclass
class LeituraTexto:
    sintoma: str = 'nenhum'
    termo_sintoma: str | None = None
    urgencia: bool = False
    violencia_relatada: bool = False
    impacto_social: bool = False


def ler_texto(texto) -> LeituraTexto:
    t = sem_acento(texto)
    out = LeituraTexto()
    if not t:
        return out
    for nivel, padroes in (('grave', TERMOS_GRAVE), ('moderado', TERMOS_MODERADO), ('leve', TERMOS_LEVE)):
        achou = _encontrou(t, padroes)
        if achou:
            out.sintoma, out.termo_sintoma = nivel, achou
            break
    out.urgencia = _encontrou(t, TERMOS_URGENCIA) is not None
    out.violencia_relatada = _encontrou(t, TERMOS_VIOLENCIA) is not None
    out.impacto_social = _encontrou(t, TERMOS_IMPACTO) is not None
    return out


# ============================================================================
# Entrada canônica
# ============================================================================
@dataclass
class Caso:
    programa: str
    idade: int
    sexo: str
    renda: str
    necessidade: str
    vitima: str
    sintoma: str
    encaminhamento: str
    tempo_espera: int
    impacto_social: bool = False
    termo_sintoma: str | None = None
    urgencia: bool = False


def montar_caso(d: dict) -> Caso:
    """Recebe o dicionário já validado (ver app.py) e devolve o caso canônico."""
    necessidade = normalizar_necessidade(d.get('necessidade'))
    leitura = ler_texto(d.get('texto'))

    # Sintoma: o mais grave entre o informado e o lido no texto
    informado = d.get('sintoma') if d.get('sintoma') in SINTOMAS else 'nenhum'
    sintoma = max(informado, leitura.sintoma, key=SINTOMAS.index)

    vitima = normalizar_vitima(d.get('vitima_violencia'), necessidade)
    if vitima == 'nao_informado' and leitura.violencia_relatada:
        vitima = 'sim'

    sexo = normalizar_sexo(d.get('sexo')) or ''
    programa = normalizar_programa(d.get('programa'))
    if programa is None:
        programa = 'apolonias_do_bem' if sexo == 'feminino' else 'dentista_do_bem'

    return Caso(
        programa=programa,
        idade=int(d.get('idade') or 0),
        sexo=sexo or ('feminino' if programa == 'apolonias_do_bem' else 'masculino'),
        renda=normalizar_renda(d.get('renda_familiar')),
        necessidade=necessidade,
        vitima=vitima,
        sintoma=sintoma,
        encaminhamento=normalizar_encaminhamento(d.get('como_soube') or d.get('encaminhamento')),
        tempo_espera=max(0, int(d.get('tempo_espera') or 0)),
        impacto_social=leitura.impacto_social,
        termo_sintoma=leitura.termo_sintoma,
        urgencia=leitura.urgencia or bool(d.get('urgencia')),
    )


# ============================================================================
# Pontuação
# ============================================================================
PONTOS_RENDA = {'ate_1': 30, '1_a_2': 20, '2_a_3': 8, 'acima_3': 0, 'nao_informada': 12}
PONTOS_NECESSIDADE = {
    'canal': 25, 'extracao': 22, 'restauracao': 14, 'outro': 8,
    'ortodontia': 6, 'limpeza': 3, 'violencia': 0, 'triagem': 0,
}
PONTOS_SINTOMA = {'nenhum': 0, 'leve': 3, 'moderado': 12, 'grave': 25}
ROTULO_RENDA = {
    'ate_1': 'até 1 salário mínimo', '1_a_2': '1 a 2 salários mínimos',
    '2_a_3': '2 a 3 salários mínimos', 'acima_3': 'acima de 3 salários mínimos',
    'nao_informada': 'renda não informada',
}
ROTULO_NECESSIDADE = {
    'canal': 'tratamento de canal', 'extracao': 'extração', 'restauracao': 'restauração',
    'ortodontia': 'ortodontia', 'limpeza': 'limpeza/prevenção', 'outro': 'outra necessidade',
    'violencia': 'dentição afetada por violência', 'triagem': 'triagem odontológica',
}


def pontos_idade_jovem(idade: int) -> int:
    """Janela do programa: quanto mais perto dos 18, menos tempo para tratar."""
    if idade < IDADE_MIN_JOVEM or idade > IDADE_MAX_JOVEM:
        return 0
    return {17: 15, 16: 12, 15: 8, 14: 5}.get(idade, 3)


def pontos_idade_mulher(idade: int) -> int:
    if idade < 18:
        return 12
    if idade >= 60:
        return 10
    if idade >= 50:
        return 8
    if idade >= 40:
        return 4
    return 0


def pontos_espera(dias: int) -> int:
    """2 pontos por semana na fila, até 15."""
    return min(15, (max(0, dias) // 7) * 2)


@dataclass
class Resultado:
    pontos: int
    prioridade: str
    regra_critica: str | None
    fatores: list[tuple[str, int]] = field(default_factory=list)
    alertas: list[str] = field(default_factory=list)


def _faixa(pontos: int) -> str:
    if pontos >= LIMIAR_ALTA:
        return 'ALTA'
    if pontos >= LIMIAR_MEDIA:
        return 'MEDIA'
    return 'BAIXA'


def avaliar(c: Caso) -> Resultado:
    fatores: list[tuple[str, int]] = []
    alertas: list[str] = []
    regra: str | None = None

    def soma(texto: str, pts: int):
        if pts:
            fatores.append((texto, pts))

    # ---------------- Comum ----------------
    soma(f'Sintoma {c.sintoma}' + (f' ("{c.termo_sintoma}")' if c.termo_sintoma else ''), PONTOS_SINTOMA[c.sintoma])
    soma(f'{c.tempo_espera} dias na fila', pontos_espera(c.tempo_espera))
    if c.impacto_social:
        soma('Impacto social relatado (autoestima, escola ou trabalho)', 5)

    if c.programa == 'dentista_do_bem':
        soma(f'Renda: {ROTULO_RENDA[c.renda]}', PONTOS_RENDA[c.renda])
        soma(f'{c.idade} anos (janela do programa até 17)', pontos_idade_jovem(c.idade))
        soma(f'Necessidade: {ROTULO_NECESSIDADE[c.necessidade]}', PONTOS_NECESSIDADE[c.necessidade])
        if c.encaminhamento == 'rede_saude_escola':
            soma('Encaminhado por posto de saúde ou escola', 4)
        if c.encaminhamento == 'rede_protecao':
            soma('Encaminhado pela rede de proteção (CREAS/Conselho Tutelar)', 8)
        if c.vitima == 'sim':
            soma('Relato de violência', 10)

        if c.idade < IDADE_MIN_JOVEM or c.idade > IDADE_MAX_JOVEM:
            alertas.append(f'Idade {c.idade} fora da faixa do Dentista do Bem (11 a 17) — confirmar elegibilidade')

        baixa_renda = c.renda in ('ate_1', '1_a_2')
        if c.renda == 'ate_1' and c.necessidade in ('canal', 'extracao'):
            regra = 'Renda até 1 salário mínimo com necessidade de canal ou extração'
        elif c.idade == 17 and baixa_renda and c.necessidade in ('canal', 'extracao', 'restauracao'):
            regra = 'Último ano no programa (17 anos), baixa renda e tratamento pendente'
        elif c.vitima == 'sim' and IDADE_MIN_JOVEM <= c.idade <= IDADE_MAX_JOVEM:
            regra = 'Adolescente com relato de violência'
        elif baixa_renda and c.necessidade in ('canal', 'extracao') and c.sintoma in ('moderado', 'grave'):
            regra = 'Baixa renda com dor e necessidade de canal ou extração'

    else:  # Apolônias do Bem
        pts_vit = {'sim': 40, 'nao_informado': 15, 'nao': 0}[c.vitima]
        soma({'sim': 'Vítima de violência com dentição afetada',
              'nao_informado': 'Violência não informada',
              'nao': ''}[c.vitima], pts_vit)
        soma(f'{c.idade} anos', pontos_idade_mulher(c.idade))
        if c.necessidade in ('canal', 'extracao'):
            soma(f'Necessidade: {ROTULO_NECESSIDADE[c.necessidade]}', 15)
        elif c.necessidade == 'restauracao':
            soma(f'Necessidade: {ROTULO_NECESSIDADE[c.necessidade]}', 8)
        if c.renda != 'nao_informada':
            soma(f'Renda: {ROTULO_RENDA[c.renda]}', PONTOS_RENDA[c.renda] // 2)
        if c.encaminhamento == 'rede_protecao' and c.vitima != 'nao':
            soma('Encaminhada por Delegacia/CREAS', 10)
        elif c.encaminhamento == 'rede_saude_escola':
            soma('Encaminhada por posto de saúde', 4)

        if c.vitima == 'nao':
            alertas.append('Sem relato de violência — Apolônias do Bem atende vítimas; confirmar elegibilidade')
        if c.vitima == 'sim' and c.idade < 18:
            regra = 'Menor de idade vítima de violência'
        elif c.vitima == 'sim' and c.sintoma == 'grave':
            regra = 'Vítima de violência com dano dentário grave (dente quebrado, perdido ou dor forte)'
        elif c.vitima == 'sim' and c.encaminhamento == 'rede_protecao' and c.sintoma == 'moderado':
            regra = 'Vítima encaminhada por Delegacia/CREAS com dor ou dano relatado'

    # Urgência clínica vale para os dois programas
    if c.urgencia and regra is None:
        regra = 'Urgência clínica (sinal de infecção, inchaço, febre, sangramento ou dor que impede comer/dormir)'

    pontos = int(min(100, sum(p for _, p in fatores)))
    prioridade = 'ALTA' if regra else _faixa(pontos)

    # Fora do perfil não sobe para ALTA (a não ser por urgência clínica)
    fora = any('elegibilidade' in a for a in alertas)
    if fora and prioridade == 'ALTA' and not c.urgencia:
        prioridade = 'MEDIA'
        regra = None

    fatores.sort(key=lambda x: -x[1])
    return Resultado(pontos=pontos, prioridade=prioridade, regra_critica=regra, fatores=fatores, alertas=alertas)
