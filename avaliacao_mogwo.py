"""Avaliação externa das cinco soluções da última execução válida do MOGWO.

Protocolo preservado da otimização:
  * CSVs do CIC-IDS2017 (MachineLearningCVE), em ordem alfabética;
  * remover linhas com NaN/infinito, depois duplicatas;
  * BENIGN = 0, demais rótulos = 1;
  * divisão estratificada 70/30, random_state=42;
  * MinMaxScaler ajustado exclusivamente no treinamento;
  * k-NN com k=3, distância euclidiana;
  * modelos treinados nos 70% de treinamento COMPLETOS;
  * máscaras FIXAS obtidas no arquivo externo do MOGWO.

A escolha de k e a otimização NÃO são refeitas. O teste externo NÃO
é usado para escolher ou modificar máscaras; todas são relatadas.

Uso:
    python -u avaliacao_mogwo.py
    python -u avaliacao_mogwo.py --solucoes B,E   # opcional: apenas dois extremos

Atenção: o k-NN treinado em 1.764.558 registros e avaliado em
756.240 registros pode exigir bastante processamento. O script salva
cada solução concluída em CSV e retoma as já verificadas.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


COLUNA_CLASSE = ' Label'
NOMES_SOLUCOES = 'ABCDE'


def carregar_e_preprocessar(pasta: Path) -> pd.DataFrame:
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de CSVs não encontrada: {pasta}")

    arquivos = sorted(pasta.glob('*.csv'))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .csv encontrado em: {pasta}")

    partes = []
    for caminho in arquivos:
        df_parte = pd.read_csv(caminho, low_memory=False)
        print(f'{caminho.name}: {df_parte.shape}', flush=True)
        partes.append(df_parte)

    df = pd.concat(partes, ignore_index=True)
    del partes
    if COLUNA_CLASSE not in df.columns:
        raise ValueError(f'Coluna {COLUNA_CLASSE!r} não encontrada no dataset.')

    print(f'\nDataset completo: {df.shape}', flush=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    anterior = len(df)
    df.dropna(inplace=True)
    print(f'Removidos por ausente/infinito: {anterior - len(df)}', flush=True)
    anterior = len(df)
    df.drop_duplicates(inplace=True)
    print(f'Removidos por duplicata: {anterior - len(df)}', flush=True)
    # Reproduz exatamente a codificação binária do mogwo.py.
    df[COLUNA_CLASSE] = df[COLUNA_CLASSE].apply(
        lambda valor: 0 if valor.strip() == 'BENIGN' else 1
    )
    print(f'Total após limpeza: {len(df)}', flush=True)
    print(df[COLUNA_CLASSE].value_counts(), flush=True)
    return df


def carregar_pareto(caminho: Path):
    if not caminho.exists():
        raise FileNotFoundError(
            f'Arquivo de Pareto não encontrado: {caminho}. '
            'Use o arquivo de cinco soluções fornecido, não o de 100 entradas.'
        )

    # Arquivos pickle devem ser abertos somente quando vêm de fonte confiável.
    with caminho.open('rb') as f:
        arquivo = pickle.load(f)

    if not isinstance(arquivo, list) or len(arquivo) != 5:
        raise ValueError(
            f'Esperava as CINCO soluções da execução válida; arquivo contém '
            f'{len(arquivo) if isinstance(arquivo, list) else "formato inesperado"}. '
            'Confira se escolheu o arquivo correto.'
        )

    solucoes = []
    for indice, sol in enumerate(arquivo):
        if not isinstance(sol, dict) or 'posicao' not in sol or 'custo' not in sol:
            raise ValueError('Arquivo Pareto com formato incompatível.')

        mascara = np.asarray(sol['posicao']).ravel()
        custos = np.asarray(sol['custo'], dtype=float).ravel()
        if len(custos) != 2 or not np.all(np.isfinite(custos)):
            raise ValueError('Custos inválidos na solução ' + NOMES_SOLUCOES[indice])
        if not np.all((mascara == 0) | (mascara == 1)):
            raise ValueError('Máscara não binária na solução ' + NOMES_SOLUCOES[indice])
        mascara = mascara.astype(np.uint8)
        if not np.any(mascara):
            raise ValueError('Máscara vazia na solução ' + NOMES_SOLUCOES[indice])

        solucoes.append({
            'solucao': NOMES_SOLUCOES[indice],
            'mascara': mascara,
            'mascara_texto': ''.join(str(int(x)) for x in mascara),
            'f1_validacao': float(1.0 - custos[0]),
            'fpr_validacao': float(custos[1]),
        })

    # Impede confundir esta execução com outros experimentos de mesmo tamanho.
    melhor_f1 = max(s['f1_validacao'] for s in solucoes)
    menor_fpr = min(s['fpr_validacao'] for s in solucoes)
    if not (abs(melhor_f1 - 0.99347135) < 0.00001 and
            abs(menor_fpr - 0.00168197) < 0.00001):
        raise ValueError(
            'O arquivo contém cinco soluções, mas as métricas não coincidem '
            'com a última execução válida (F1≈0,993471; FPR≈0,001682).'
        )

    print('\nSoluções fixas carregadas (validação interna):', flush=True)
    for sol in solucoes:
        print(
            f" {sol['solucao']}: "
            f"F1={sol['f1_validacao']:.6f}, "
            f"FPR={sol['fpr_validacao']:.6f}, "
            f"atributos={int(sol['mascara'].sum())}",
            flush=True
        )
    return solucoes


def selecionar_solucoes(solucoes, especificacao: str):
    if especificacao.strip().lower() == 'todas':
        return solucoes
    nomes = {p.strip().upper() for p in especificacao.split(',') if p.strip()}
    validos = {sol['solucao'] for sol in solucoes}
    if not nomes or not nomes.issubset(validos):
        raise ValueError(f'--solucoes deve ser "todas" ou letras dentre {sorted(validos)}.')
    return [sol for sol in solucoes if sol['solucao'] in nomes]


def carregar_concluidos(saida: Path, solucoes):
    if not saida.exists():
        return []
    df = pd.read_csv(saida, dtype={'mascara': str})
    if df.empty:
        return []
    if not {'solucao', 'mascara', 'f1_validacao', 'fpr_validacao'}.issubset(df.columns):
        raise ValueError(f'O CSV existente {saida} não tem as colunas esperadas.')
    gabarito = {s['solucao']: s for s in solucoes}
    for _, linha in df.iterrows():
        chave = str(linha['solucao'])
        s = gabarito.get(chave)
        if (s is None or str(linha['mascara']) != s['mascara_texto'] or
            not np.isclose(float(linha['f1_validacao']), s['f1_validacao'], atol=1e-12) or
            not np.isclose(float(linha['fpr_validacao']), s['fpr_validacao'], atol=1e-12)):
            raise ValueError(
                f'O arquivo de saída {saida} parece corresponder a outro arquivo '
                'de Pareto. Renomeie-o antes de executar novamente.'
            )
    return df.to_dict('records')


def avaliar_solucao(df: pd.DataFrame, indices_treino: np.ndarray,
                     indices_teste: np.ndarray, y: np.ndarray,
                     colunas_atributos: list[int], solucao: dict,
                     tamanho_lote: int):
    inicio = time.perf_counter()
    indices_atributos = np.flatnonzero(solucao['mascara'])
    posicoes_df = [colunas_atributos[i] for i in indices_atributos]

    print(f"\n=== Solução {solucao['solucao']} ({len(indices_atributos)} atributos) ===", flush=True)
    print('Preparando e normalizando os atributos selecionados...', flush=True)
    # MinMaxScaler é independente por coluna: ajustar apenas as colunas
    # selecionadas produz a MESMA transformação por atributo que ajustar
    # o scaler às 78 colunas, mas usa menos memória.
    X_treino = df.iloc[indices_treino, posicoes_df].to_numpy(dtype=np.float64, copy=True)
    X_teste = df.iloc[indices_teste, posicoes_df].to_numpy(dtype=np.float64, copy=True)
    scaler = MinMaxScaler(copy=False)
    scaler.fit_transform(X_treino)
    scaler.transform(X_teste)

    y_treino = y[indices_treino]
    y_teste = y[indices_teste]
    modelo = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)

    print(f'Treinando k-NN em {len(y_treino):,} registros...', flush=True)
    modelo.fit(X_treino, y_treino)
    # Libera a matriz original do dataframe apenas ao final da função.
    print(f'Classificando {len(y_teste):,} registros de teste em lotes...', flush=True)

    # Matriz de confusão acumulada; não precisamos reter todas as predições.
    tn = fp = fn = tp = 0
    n_lotes = (len(y_teste) + tamanho_lote - 1) // tamanho_lote
    for numero_lote, comeco in enumerate(range(0, len(y_teste), tamanho_lote), 1):
        fim = min(comeco + tamanho_lote, len(y_teste))
        predicoes = modelo.predict(X_teste[comeco:fim])
        a, b, c, d = confusion_matrix(y_teste[comeco:fim], predicoes, labels=[0, 1]).ravel()
        tn += int(a)
        fp += int(b)
        fn += int(c)
        tp += int(d)

        if numero_lote == 1 or numero_lote % 10 == 0 or numero_lote == n_lotes:
            decorrido = time.perf_counter() - inicio
            porcentagem = 100.0 * fim / len(y_teste)
            print(
                f'  Lote {numero_lote}/{n_lotes} | {porcentagem:.1f}% do teste '
                f'| tempo decorrido: {decorrido / 60:.1f} min',
                flush=True
            )

    precisao = tp / (tp + fp) if tp + fp else 0.0
    revocacao = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    tempo = time.perf_counter() - inicio

    print(f'\nMatriz de confusão: TN={tn}, FP={fp}, FN={fn}, TP={tp}', flush=True)
    print(f'F1 teste: {f1:.6f} | FPR teste: {fpr:.6f}', flush=True)
    print(f'Precisão: {precisao:.6f} | Revocação: {revocacao:.6f}', flush=True)
    print(f'Acurácia: {acuracia:.6f} | Tempo: {tempo / 60:.1f} min', flush=True)

    return {
        'solucao': solucao['solucao'],
        'mascara': solucao['mascara_texto'],
        'n_atributos': int(len(indices_atributos)),
        'indices_atributos_base_0': ' '.join(str(i) for i in indices_atributos),
        'nomes_atributos': ' | '.join(str(df.columns[p]) for p in posicoes_df),
        'f1_validacao': solucao['f1_validacao'],
        'fpr_validacao': solucao['fpr_validacao'],
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
        'tempo_segundos': tempo,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', default='data/MachineLearningCVE', type=Path)
    parser.add_argument('--arquivo', default='arquivo_pareto_mogwo_5solucoes.pkl', type=Path)
    parser.add_argument('--saida', default='resultados_teste_mogwo.csv', type=Path)
    parser.add_argument('--lote', default=1000, type=int,
                        help='Nº de instâncias classificadas por lote (padrão: 1000).')
    parser.add_argument('--solucoes', default='todas',
                        help='"todas" (padrão) ou letras separadas por vírgula, ex.: B,E')
    args = parser.parse_args()
    if args.lote <= 0:
        parser.error('--lote precisa ser maior que zero')

    solucoes = carregar_pareto(args.arquivo)
    escolhidas = selecionar_solucoes(solucoes, args.solucoes)
    anteriores = carregar_concluidos(args.saida, solucoes)
    concluidas = {str(r['solucao']) for r in anteriores}
    pendentes = [s for s in escolhidas if s['solucao'] not in concluidas]

    if not pendentes:
        print('\nTodas as soluções solicitadas já constam no CSV de saída.', flush=True)
        return

    df = carregar_e_preprocessar(args.dados)
    posicoes_atributos = [i for i, c in enumerate(df.columns) if c != COLUNA_CLASSE]
    n_atributos = len(posicoes_atributos)
    for sol in solucoes:
        if len(sol['mascara']) != n_atributos:
            raise ValueError(
                f"Máscara da solução {sol['solucao']} possui {len(sol['mascara'])} "
                f'posições, mas o dataset tem {n_atributos} atributos. '
                'Confira o pré-processamento e a ordem dos CSVs.'
            )

    y = df[COLUNA_CLASSE].to_numpy(dtype=np.uint8, copy=True)
    # Dividir índices preserva EXATAMENTE a divisão estratificada original
    # train_test_split(X, y, test_size=0.3, stratify=y, random_state=42).
    indices_treino, indices_teste = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=42
    )
    print(f'\nDivisão externa: {len(indices_treino):,} treino; '
          f'{len(indices_teste):,} teste (seed=42).', flush=True)
    print('Treinamento: 100% do conjunto EXTERNO de treino; não a amostra de 10%.', flush=True)
    print('O conjunto de teste só é usado para avaliação, não para alterar máscaras.', flush=True)

    resultados = anteriores[:]
    for sol in pendentes:
        resultado = avaliar_solucao(df, indices_treino, indices_teste, y,
                                    posicoes_atributos, sol, args.lote)
        resultados.append(resultado)
        resultados.sort(key=lambda linha: NOMES_SOLUCOES.index(str(linha['solucao'])))
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(resultados).to_csv(args.saida, index=False, encoding='utf-8-sig')
        print(f'Resultado salvo em {args.saida}', flush=True)

    print('\nAvaliação externa concluída. Resumo:', flush=True)
    for linha in resultados:
        print(f" {linha['solucao']}: F1={float(linha['f1_teste']):.6f} | "
              f"FPR={float(linha['fpr_teste']):.6f} | "
              f"atributos={linha['n_atributos']}", flush=True)
    print(f'Arquivo final: {args.saida}', flush=True)


if __name__ == '__main__':
    main()
