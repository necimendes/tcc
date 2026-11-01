"""Avaliacao externa das mascaras FIXAS de solucoes.csv.

Entrada: solucoes.csv, produzido por mogwo.py --selecionar (validacao interna).
Saida: teste.csv, com as metricas do conjunto externo de teste.

Nao executa MOGWO, nao seleciona mascaras e nao modifica o Pareto.
Reproduz o protocolo da otimizacao:
- mesmo CIC-IDS2017 e remocao de NaN/Inf/duplicatas;
- classes: benigno=0, ataques=1;
- divisao externa estratificada 70/30 (random_state=42);
- MinMaxScaler ajustado somente sobre o treino externo;
- k-NN com k=3 e distancia euclidiana;
- treinamento com todos os 1.764.558 registros do treino externo;
- avaliacao com os 756.240 registros de teste externo.

Uso:
    python -u avaliacao.py
Opcionalmente, para reaproveitar somente mascaras IDENTICAS ja avaliadas:
    python -u avaliacao.py --reaproveitar teste_anterior.csv

O reaproveitamento nao seleciona candidatos: apenas copia os resultados
anteriores quando a mascara e o protocolo sao exatamente os mesmos.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

COLUNA_CLASSE = ' Label'
COLUNAS_OBRIGATORIAS = {
    'solucao', 'tipo', 'mascara', 'n_atributos', 'f1_score', 'fpr',
}
CAMPOS_SAIDA = [
    'solucao', 'tipo', 'iteracao', 'mascara', 'n_atributos',
    'distancia_hamming_pareto', 'atributos_a_menos',
    'f1_validacao', 'fpr_validacao',
    'f1_teste', 'fpr_teste', 'precisao_teste', 'revocacao_teste',
    'acuracia_teste', 'tn', 'fp', 'fn', 'tp',
    'n_treino', 'n_teste', 'tempo_segundos', 'atributos', 'origem_resultado',
]


def carregar_solucoes(caminho: Path) -> list[dict]:
    if not caminho.is_file():
        raise FileNotFoundError(f'Nao encontrei {caminho}. Execute mogwo.py --selecionar primeiro.')

    dados = pd.read_csv(caminho, dtype={'mascara': str, 'solucao': str, 'tipo': str})
    faltantes = COLUNAS_OBRIGATORIAS.difference(dados.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes em {caminho}: {sorted(faltantes)}')
    if dados.empty:
        raise ValueError(f'{caminho} esta vazio.')
    if dados['solucao'].duplicated().any() or dados['mascara'].duplicated().any():
        raise ValueError('Ha IDs ou mascaras duplicadas em solucoes.csv.')

    solucoes = []
    comprimento = None
    for _, linha in dados.iterrows():
        identificador = str(linha['solucao'])
        mascara_txt = str(linha['mascara'])
        if not mascara_txt or set(mascara_txt) - {'0', '1'}:
            raise ValueError(f'Mascara nao binaria na solucao {identificador}.')
        if comprimento is None:
            comprimento = len(mascara_txt)
        if len(mascara_txt) != comprimento:
            raise ValueError('As mascaras possuem comprimentos diferentes.')
        mascara = np.fromiter((int(b) for b in mascara_txt), dtype=np.uint8)
        n_atributos = int(mascara.sum())
        if n_atributos == 0 or n_atributos != int(linha['n_atributos']):
            raise ValueError(f'Quantidade de atributos inconsistente em {identificador}.')
        f1 = float(linha['f1_score'])
        fpr = float(linha['fpr'])
        if not (np.isfinite(f1) and 0 <= f1 <= 1 and np.isfinite(fpr) and 0 <= fpr <= 1):
            raise ValueError(f'Metricas de validacao invalidas em {identificador}.')
        solucoes.append({
            'solucao': identificador,
            'tipo': str(linha['tipo']),
            'iteracao': int(linha['iteracao']) if 'iteracao' in dados.columns else None,
            'mascara': mascara,
            'mascara_txt': mascara_txt,
            'n_atributos': n_atributos,
            'distancia_hamming_pareto': (
                int(linha['distancia_hamming_pareto'])
                if 'distancia_hamming_pareto' in dados.columns else None
            ),
            'atributos_a_menos': (
                int(linha['atributos_a_menos'])
                if 'atributos_a_menos' in dados.columns else None
            ),
            'f1_validacao': f1,
            'fpr_validacao': fpr,
        })
    return solucoes


def validar_linha_preexistente(linha: dict, sol: dict) -> None:
    if (str(linha['mascara']) != sol['mascara_txt']
        or str(linha['tipo']) != sol['tipo']
        or int(linha['n_atributos']) != sol['n_atributos']
        or not np.isclose(float(linha['f1_validacao']), sol['f1_validacao'], atol=1e-12, rtol=0)
        or not np.isclose(float(linha['fpr_validacao']), sol['fpr_validacao'], atol=1e-12, rtol=0)):
        raise ValueError('teste.csv corresponde a outra selecao; renomeie-o antes da nova avaliacao.')


def carregar_resultados_anteriores(caminho: Path, solucoes: list[dict]) -> list[dict]:
    if not caminho.exists():
        return []
    tabela = pd.read_csv(caminho, dtype={'mascara': str, 'solucao': str})
    if tabela.empty:
        return []
    faltantes = set(CAMPOS_SAIDA).difference(tabela.columns)
    if faltantes:
        raise ValueError(
            f'{caminho} ja existe mas tem colunas incompativeis: {sorted(faltantes)}. '
            'Guarde o arquivo anterior com outro nome antes de continuar.'
        )
    referencias = {s['solucao']: s for s in solucoes}
    if tabela['solucao'].duplicated().any():
        raise ValueError(f'IDs duplicados em {caminho}.')
    for linha in tabela.to_dict('records'):
        sol = referencias.get(str(linha['solucao']))
        if sol is None:
            raise ValueError(f'{caminho} contem uma solucao que nao esta na nova selecao.')
        validar_linha_preexistente(linha, sol)
    return tabela.to_dict('records')


def montar_cabecalho(sol: dict) -> dict:
    return {
        'solucao': sol['solucao'],
        'tipo': sol['tipo'],
        'iteracao': sol['iteracao'],
        'mascara': sol['mascara_txt'],
        'n_atributos': sol['n_atributos'],
        'distancia_hamming_pareto': sol['distancia_hamming_pareto'],
        'atributos_a_menos': sol['atributos_a_menos'],
        'f1_validacao': sol['f1_validacao'],
        'fpr_validacao': sol['fpr_validacao'],
    }


def salvar_resultados(caminho: Path, resultados: list[dict], solucoes: list[dict]) -> None:
    ordem = {s['solucao']: i for i, s in enumerate(solucoes)}
    resultados_ordenados = sorted(resultados, key=lambda r: ordem[r['solucao']])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + '.tmp')
    pd.DataFrame(resultados_ordenados, columns=CAMPOS_SAIDA).to_csv(
        temporario, index=False, encoding='utf-8-sig'
    )
    os.replace(temporario, caminho)


def reaproveitar_resultados(arquivo_antigo: Path, solucoes: list[dict],
                            resultados: list[dict]) -> int:
    if not arquivo_antigo.is_file():
        raise FileNotFoundError(f'Nao encontrei {arquivo_antigo}.')
    antigo = pd.read_csv(arquivo_antigo, dtype={'mascara': str})
    exigidas = {
        'mascara', 'n_atributos', 'f1_teste', 'fpr_teste', 'precisao_teste',
        'revocacao_teste', 'acuracia_teste', 'tn', 'fp', 'fn', 'tp',
        'n_treino', 'n_teste', 'tempo_segundos', 'atributos',
    }
    faltantes = exigidas.difference(antigo.columns)
    if faltantes:
        raise ValueError(f'Arquivo para reaproveitamento tem colunas ausentes: {sorted(faltantes)}')
    if antigo['mascara'].duplicated().any():
        raise ValueError('Arquivo de teste antigo contem mascaras repetidas.')
    por_mascara = antigo.set_index('mascara', drop=False)
    concluidas = {r['solucao'] for r in resultados}
    quantidade = 0
    for sol in solucoes:
        if sol['solucao'] in concluidas or sol['mascara_txt'] not in por_mascara.index:
            continue
        antiga = por_mascara.loc[sol['mascara_txt']]
        if int(antiga['n_atributos']) != sol['n_atributos']:
            raise ValueError(f'Quantidade de atributos difere no teste antigo: {sol["solucao"]}')
        tn, fp, fn, tp = (int(antiga[col]) for col in ('tn', 'fp', 'fn', 'tp'))
        if (tn + fp + fn + tp != int(antiga['n_teste'])
            or int(antiga['n_treino']) != 1764558
            or int(antiga['n_teste']) != 756240):
            raise ValueError('Contagens de treino/teste antigas incompativeis com esta execucao.')
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        if (not np.isclose(f1, float(antiga['f1_teste']), atol=1e-10)
            or not np.isclose(fpr, float(antiga['fpr_teste']), atol=1e-10)):
            raise ValueError('Metricas antigas nao correspondem a matriz de confusao.')
        registro = montar_cabecalho(sol)
        for col in (
            'f1_teste', 'fpr_teste', 'precisao_teste', 'revocacao_teste',
            'acuracia_teste', 'tn', 'fp', 'fn', 'tp',
            'n_treino', 'n_teste', 'tempo_segundos', 'atributos',
        ):
            registro[col] = antiga[col]
        registro['origem_resultado'] = 'reaproveitado_teste_anterior'
        resultados.append(registro)
        concluidas.add(sol['solucao'])
        quantidade += 1
    return quantidade


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.is_dir():
        raise FileNotFoundError(f'Pasta nao encontrada: {caminho}')
    arquivos = sorted(caminho.glob('*.csv'))
    if not arquivos:
        raise FileNotFoundError(f'Nao ha arquivos CSV em {caminho}')

    partes = []
    for arquivo in arquivos:
        df_parte = pd.read_csv(arquivo, low_memory=False)
        print(f'{arquivo.name}: {df_parte.shape}', flush=True)
        partes.append(df_parte)
    df = pd.concat(partes, ignore_index=True)
    del partes
    if COLUNA_CLASSE not in df.columns:
        raise ValueError(f'Coluna {COLUNA_CLASSE!r} nao encontrada no dataset.')
    print(f'\nDataset completo: {df.shape}', flush=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    antes = len(df)
    df.dropna(inplace=True)
    print(f'Removidos por ausente/infinito: {antes - len(df)}', flush=True)
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f'Removidos por duplicata: {antes - len(df)}', flush=True)
    df[COLUNA_CLASSE] = df[COLUNA_CLASSE].apply(
        lambda x: 0 if x.strip() == 'BENIGN' else 1
    )
    print(f'Total apos limpeza: {len(df)}', flush=True)
    print(df[COLUNA_CLASSE].value_counts(), flush=True)
    return df


def avaliar_uma(df: pd.DataFrame, colunas: list[int], y: np.ndarray,
                 indices_treino: np.ndarray, indices_teste: np.ndarray,
                 sol: dict, tamanho_lote: int) -> dict:
    inicio = time.perf_counter()
    indices_selecionados = np.flatnonzero(sol['mascara'])
    posicoes_df = [colunas[i] for i in indices_selecionados]
    y_treino = y[indices_treino]
    y_teste = y[indices_teste]

    print(f"\n{sol['solucao']} [{sol['tipo']}] | "
          f"{sol['n_atributos']} atributos", flush=True)
    X_treino = df.iloc[indices_treino, posicoes_df].to_numpy(dtype=np.float64, copy=True)
    escalonador = MinMaxScaler(copy=False)
    X_treino = escalonador.fit_transform(X_treino)
    modelo = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
    modelo.fit(X_treino, y_treino)

    tn = fp = fn = tp = 0
    n_lotes = (len(indices_teste) + tamanho_lote - 1) // tamanho_lote
    for n, inicio_lote in enumerate(range(0, len(indices_teste), tamanho_lote), 1):
        fim_lote = min(inicio_lote + tamanho_lote, len(indices_teste))
        posicoes_lote = indices_teste[inicio_lote:fim_lote]
        X_lote = df.iloc[posicoes_lote, posicoes_df].to_numpy(dtype=np.float64, copy=True)
        X_lote = escalonador.transform(X_lote)
        previstos = modelo.predict(X_lote)
        verdadeiros = y_teste[inicio_lote:fim_lote]
        tn += int(np.count_nonzero((verdadeiros == 0) & (previstos == 0)))
        fp += int(np.count_nonzero((verdadeiros == 0) & (previstos == 1)))
        fn += int(np.count_nonzero((verdadeiros == 1) & (previstos == 0)))
        tp += int(np.count_nonzero((verdadeiros == 1) & (previstos == 1)))
        if n == 1 or n % 50 == 0 or n == n_lotes:
            print(f'Teste {fim_lote:,}/{len(indices_teste):,} '
                  f'({100 * fim_lote / len(indices_teste):.1f}%)', flush=True)

    precisao = tp / (tp + fp) if tp + fp else 0.0
    revocacao = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    decorrido = time.perf_counter() - inicio
    print(f'F1={f1:.6f} | FPR={fpr:.6f} | acuracia={acuracia:.6f} | '
          f'TN={tn}, FP={fp}, FN={fn}, TP={tp}', flush=True)

    return {
        **montar_cabecalho(sol),
        'f1_teste': f1,
        'fpr_teste': fpr,
        'precisao_teste': precisao,
        'revocacao_teste': revocacao,
        'acuracia_teste': acuracia,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'n_treino': len(y_treino),
        'n_teste': len(y_teste),
        'tempo_segundos': decorrido,
        'atributos': ' | '.join(str(df.columns[pos]) for pos in posicoes_df),
        'origem_resultado': 'avaliado_nesta_execucao',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=Path('data/MachineLearningCVE'))
    parser.add_argument('--solucoes', type=Path, default=Path('solucoes.csv'))
    parser.add_argument('--saida', type=Path, default=Path('teste.csv'))
    parser.add_argument('--lote', type=int, default=1000)
    parser.add_argument('--reaproveitar', type=Path, default=None,
                        help='CSV de teste anterior para copiar somente mascaras identicas')
    args = parser.parse_args()
    if args.lote <= 0:
        parser.error('--lote deve ser maior que zero')
    if args.reaproveitar and args.reaproveitar.resolve() == args.saida.resolve():
        parser.error('--reaproveitar e --saida devem indicar arquivos diferentes')

    solucoes = carregar_solucoes(args.solucoes)
    resultados = carregar_resultados_anteriores(args.saida, solucoes)
    concluidas = {r['solucao'] for r in resultados}

    if args.reaproveitar:
        reaproveitadas = reaproveitar_resultados(args.reaproveitar, solucoes, resultados)
        if reaproveitadas:
            salvar_resultados(args.saida, resultados, solucoes)
        print(f'Resultados antigos aproveitados por mascara identica: {reaproveitadas}', flush=True)
        concluidas = {r['solucao'] for r in resultados}

    pendentes = [s for s in solucoes if s['solucao'] not in concluidas]
    print(f'Solucoes selecionadas: {len(solucoes)} | '
          f'Concluidas: {len(concluidas)} | Pendentes: {len(pendentes)}', flush=True)
    if not pendentes:
        print(f'Avaliacao completa em {args.saida}.', flush=True)
        return

    df = carregar_dataset(args.dados)
    colunas = [i for i, col in enumerate(df.columns) if col != COLUNA_CLASSE]
    if len(solucoes[0]['mascara']) != len(colunas):
        raise ValueError(f'Mascaras de {len(solucoes[0]["mascara"])} bits, '
                         f'mas dataset com {len(colunas)} atributos.')
    y = df[COLUNA_CLASSE].to_numpy(dtype=np.uint8, copy=True)
    indices_treino, indices_teste = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=42
    )
    print(f'Treino externo: {len(indices_treino):,}; '
          f'teste externo: {len(indices_teste):,}', flush=True)
    print('O teste nao interfere na selecao das mascaras.', flush=True)

    for sol in pendentes:
        resultado = avaliar_uma(df, colunas, y, indices_treino, indices_teste,
                               sol, args.lote)
        resultados.append(resultado)
        salvar_resultados(args.saida, resultados, solucoes)
        print(f'Salvo em {args.saida}: {len(resultados)}/{len(solucoes)}', flush=True)

    print(f'\nAvaliacao concluida: {len(resultados)} solucoes em {args.saida}.', flush=True)


if __name__ == '__main__':
    main()
