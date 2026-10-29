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
COLUNAS_ENTRADA = {
    'solucao', 'tipo', 'mascara', 'n_atributos', 'f1_score', 'fpr'
}


def carregar_solucoes(caminho: Path) -> list[dict]:
   
    dados = pd.read_csv(caminho, dtype={'mascara': str, 'solucao': str, 'tipo': str})
    faltantes = COLUNAS_ENTRADA.difference(dados.columns)
    if faltantes:
        raise ValueError(f'colunas ausentes em {caminho}: {sorted(faltantes)}')
    if dados.empty:
        raise ValueError('solucoes.csv está vazio.')
    if dados['solucao'].duplicated().any() or dados['mascara'].duplicated().any():
        raise ValueError('há ids ou máscaras repetidos em solucoes.csv.')

    solucoes = []
    comprimento = None
    for _, linha in dados.iterrows():
        identificador = str(linha['solucao'])
        mascara_str = str(linha['mascara'])
        if not mascara_str or set(mascara_str) - {'0', '1'}:
            raise ValueError(f'máscara não binária: {identificador}.')
        if comprimento is None:
            comprimento = len(mascara_str)
        if len(mascara_str) != comprimento:
            raise ValueError('as máscaras não possuem o mesmo comprimento.')
        mascara = np.fromiter((int(bit) for bit in mascara_str), dtype=np.uint8)
        numero_atributos = int(mascara.sum())
        if numero_atributos == 0 or numero_atributos != int(linha['n_atributos']):
            raise ValueError(f'quantidade de atributos inconsistente: {identificador}.')
        f1_val = float(linha['f1_score'])
        fpr_val = float(linha['fpr'])
        if not (np.isfinite(f1_val) and np.isfinite(fpr_val)):
            raise ValueError(f'métricas inválidas: {identificador}.')
        solucoes.append({
            'solucao': identificador,
            'tipo': str(linha['tipo']),
            'mascara': mascara,
            'mascara_texto': mascara_str,
            'n_atributos': numero_atributos,
            'f1_validacao': f1_val,
            'fpr_validacao': fpr_val,
        })
    return solucoes


def carregar_resultados_anteriores(caminho: Path, solucoes: list[dict]) -> list[dict]:
    if not caminho.exists():
        return []
    tabela = pd.read_csv(caminho, dtype={'mascara': str, 'solucao': str})
    if tabela.empty:
        return []
    exigidas = {'solucao', 'mascara', 'tipo', 'f1_validacao', 'fpr_validacao',
                'f1_teste', 'fpr_teste'}
    if exigidas.difference(tabela.columns):
        raise ValueError(f'{caminho} não tem as colunas esperadas; não vou sobrescrevê-lo.')
    referencias = {s['solucao']: s for s in solucoes}
    if tabela['solucao'].duplicated().any():
        raise ValueError(f'ids duplicados em {caminho}.')
    for _, linha in tabela.iterrows():
        sol = referencias.get(str(linha['solucao']))
        if (sol is None
            or str(linha['mascara']) != sol['mascara_texto']
            or str(linha['tipo']) != sol['tipo']
            or not np.isclose(float(linha['f1_validacao']), sol['f1_validacao'], atol=1e-12, rtol=0)
            or not np.isclose(float(linha['fpr_validacao']), sol['fpr_validacao'], atol=1e-12, rtol=0)):
            raise ValueError(
                f'{caminho} corresponde a outra seleção.'
            )
    return tabela.to_dict('records')


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    if not caminho.is_dir():
        raise FileNotFoundError(f'pasta não encontrada: {caminho}')
    arquivos = sorted(caminho.glob('*.csv'))
    if not arquivos:
        raise FileNotFoundError(f'não há csvs em {caminho}')

    partes = []
    for arquivo in arquivos:
        df_parte = pd.read_csv(arquivo, low_memory=False)
        print(f'{arquivo.name}: {df_parte.shape}', flush=True)
        partes.append(df_parte)
    df = pd.concat(partes, ignore_index=True)
    del partes
    if COLUNA_CLASSE not in df.columns:
        raise ValueError(f'coluna {COLUNA_CLASSE!r} não encontrada.')

    print(f'\ndataset completo: {df.shape}', flush=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    antes = len(df)
    df.dropna(inplace=True)
    print(f'removidos por ausente/infinito: {antes - len(df)}', flush=True)
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f'removidos por duplicata: {antes - len(df)}', flush=True)
    df[COLUNA_CLASSE] = df[COLUNA_CLASSE].apply(
        lambda x: 0 if x.strip() == 'BENIGN' else 1
    )
    print(f'total após limpeza: {len(df)}', flush=True)
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

    print(f"\nsolução {sol['solucao']} [{sol['tipo']}] — "
          f"{sol['n_atributos']} atributos", flush=True)
   
    X_treino = df.iloc[indices_treino, posicoes_df].to_numpy(dtype=np.float64, copy=True)
    scaler = MinMaxScaler(copy=False)
    X_treino = scaler.fit_transform(X_treino)

    modelo = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
    print(f'treinando knn em {len(y_treino):,} amostras', flush=True)
    modelo.fit(X_treino, y_treino)

    tn = fp = fn = tp = 0
    n_lotes = (len(indices_teste) + tamanho_lote - 1) // tamanho_lote
    for numero_lote, inicio_lote in enumerate(range(0, len(indices_teste), tamanho_lote), 1):
        fim_lote = min(inicio_lote + tamanho_lote, len(indices_teste))
        lote_indices = indices_teste[inicio_lote:fim_lote]
        X_lote = df.iloc[lote_indices, posicoes_df].to_numpy(dtype=np.float64, copy=True)
        X_lote = scaler.transform(X_lote)
        previstos = modelo.predict(X_lote)
        verdadeiros = y_teste[inicio_lote:fim_lote]
        tn += int(np.count_nonzero((verdadeiros == 0) & (previstos == 0)))
        fp += int(np.count_nonzero((verdadeiros == 0) & (previstos == 1)))
        fn += int(np.count_nonzero((verdadeiros == 1) & (previstos == 0)))
        tp += int(np.count_nonzero((verdadeiros == 1) & (previstos == 1)))

        if numero_lote == 1 or numero_lote % 50 == 0 or numero_lote == n_lotes:
            print(f"  teste {fim_lote:,}/{len(indices_teste):,} "
                  f"({100 * fim_lote / len(indices_teste):.1f}%)", flush=True)

    precisao = tp / (tp + fp) if tp + fp else 0.0
    revocacao = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    decorrido = time.perf_counter() - inicio

    print(f"  f1 teste: {f1:.6f} | fpr teste: {fpr:.6f} | "
          f"precisão: {precisao:.6f} | revocação: {revocacao:.6f}", flush=True)
    print(f'  matriz: tn={tn} FP={fp} fn={fn} tp={tp}; '
          f'tempo {decorrido/60:.1f} min', flush=True)

    return {
        'solucao': sol['solucao'],
        'tipo': sol['tipo'],
        'mascara': sol['mascara_texto'],
        'n_atributos': sol['n_atributos'],
        'f1_validacao': sol['f1_validacao'],
        'fpr_validacao': sol['fpr_validacao'],
        'f1_teste': f1,
        'fpr_teste': fpr,
        'precisao_teste': precisao,
        'revocacao_teste': revocacao,
        'acuracia_teste': acuracia,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'n_treino': len(y_treino),
        'n_teste': len(y_teste),
        'tempo_segundos': decorrido,
        'atributos': ' | '.join(str(df.columns[pos]) for pos in posicoes_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=Path('data/MachineLearningCVE'))
    parser.add_argument('--solucoes', type=Path, default=Path('solucoes.csv'))
    parser.add_argument('--saida', type=Path, default=Path('teste.csv'))
    parser.add_argument('--lote', type=int, default=1000)
    args = parser.parse_args()
    if args.lote <= 0:
        parser.error('--lote deve ser maior que zero')

    solucoes = carregar_solucoes(args.solucoes)
    resultados = carregar_resultados_anteriores(args.saida, solucoes)
    concluidas = {r['solucao'] for r in resultados}
    pendentes = [s for s in solucoes if s['solucao'] not in concluidas]
    print(f'{len(solucoes)} soluções no csv: {len(concluidas)} concluídas, '
          f'{len(pendentes)} pendentes.', flush=True)
    if not pendentes:
        print(f'todas já constam em {args.saida}.', flush=True)
        return

    df = carregar_dataset(args.dados)
    colunas = [i for i, coluna in enumerate(df.columns) if coluna != COLUNA_CLASSE]
    if len(solucoes[0]['mascara']) != len(colunas):
        raise ValueError(
            f'Máscaras têm {len(solucoes[0]["mascara"])} bits; '
            f'dataset possui {len(colunas)} atributos. Confira os dados.'
        )

    y = df[COLUNA_CLASSE].to_numpy(dtype=np.uint8, copy=True)
    indices_treino, indices_teste = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=42
    )
    print(f'\ntreino externo: {len(indices_treino):,}; '
          f'teste externo: {len(indices_teste):,}.', flush=True)

    ordem = {s['solucao']: i for i, s in enumerate(solucoes)}
    for sol in pendentes:
        resultado = avaliar_uma(df, colunas, y, indices_treino, indices_teste,
                               sol, args.lote)
        resultados.append(resultado)
        resultados.sort(key=lambda r: ordem[r['solucao']])
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        temporario = args.saida.with_name(args.saida.name + '.tmp')
        pd.DataFrame(resultados).to_csv(temporario, index=False, encoding='utf-8-sig')
        os.replace(temporario, args.saida)
        print(f"  salvo em {args.saida} ({len(resultados)}/{len(solucoes)}).", flush=True)

    print(f'\navaliação externa completa: {len(resultados)} soluções em {args.saida}.', flush=True)


if __name__ == '__main__':
    main()
