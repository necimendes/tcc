# MOGWO com dois objetivos: maximizar F1-score e minimizar FPR.
# Uma selecao auxiliar de solucoes de alto desempenho e feita SOMENTE
# depois da otimizacao; ela nao interfere no algoritmo nem na dominancia.
# "avaliacoes.csv" e "solucoes.csv" usam VALIDACAO INTERNA, nao teste externo.

import numpy as np
import pandas as pd
import os
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pickle
import sys

def carregar_dataset(caminho):
    dfs = []
    for arquivo in sorted(os.listdir(caminho)):
        if arquivo.endswith('.csv'):
            df_temp = pd.read_csv(os.path.join(caminho, arquivo), low_memory=False)
            print(f"{arquivo}: {df_temp.shape}")
            dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    print(f"\ndataset completo: {df.shape}")
    return df

def preprocessar(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    antes = len(df)
    df.dropna(inplace=True)
    print(f"removidos por ausente/infinito: {antes - len(df)}")
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f"removidos por duplicata: {antes - len(df)}")
    df[' Label'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
    print(f"total após limpeza: {len(df)}")
    print(df[' Label'].value_counts())
    return df

def dividir_normalizar(df):
    X = df.drop(columns=[' Label'])
    y = df[' Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"treino: {X_train_scaled.shape[0]} | teste: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test

def criar_amostra(X_train_scaled, y_train, tamanho=0.10):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-tamanho, random_state=42)
    idx, _ = next(sss.split(X_train_scaled, y_train))
    X_amostra = X_train_scaled[idx]
    y_amostra = y_train.iloc[idx].reset_index(drop=True)
    print(f"amostra: {len(X_amostra)} registros")
    return X_amostra, y_amostra

def escolher_k(X_amostra, y_amostra, ks=[3, 5, 7, 9]):
    resultados = {}
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X_amostra, y_amostra, cv=cv, scoring='f1', n_jobs=-1)
        resultados[k] = scores
        print(f"k={k}: F1 médio={scores.mean():.4f} | desvio={scores.std():.4f}")
    k_escolhido = max(resultados, key=lambda k: resultados[k].mean())
    print(f"\nk escolhido: {k_escolhido}")
    return k_escolhido


from sklearn.metrics import confusion_matrix
import csv
import copy


def domina(a, b):
    """Dominância de Pareto: minimiza [1 - F1, FPR]."""
    return bool(np.all(a <= b) and np.any(a < b))


def avaliar_subconjunto(posicao, X_tr, X_val, y_tr, y_val, k):
    indices = np.flatnonzero(posicao)
    if len(indices) == 0:
        return np.array([1.0, 1.0])
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr[:, indices], y_tr)
    pred = knn.predict(X_val[:, indices])
    f1 = f1_score(y_val, pred, zero_division=0)
    tn, fp, _, _ = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    return np.array([
        1.0 - f1,
        fp / (fp + tn) if fp + tn else 0.0
    ])


def criar_hipercubos(arquivo, n_grid=10, alpha=0.1):
    """identifica o hipercubo de cada solução no espaço dos objetivos"""
    custos = np.array([sol['custo'] for sol in arquivo])
    limites_por_objetivo = []

    for j in range(custos.shape[1]):
        minimo, maximo = custos[:, j].min(), custos[:, j].max()
        margem = alpha * (maximo - minimo)

        limites_por_objetivo.append(
            np.linspace(minimo - margem, maximo + margem, n_grid - 1)
        )

    return [
        tuple(
            int(np.searchsorted(limites, custo[j], side='left'))
            for j, limites in enumerate(limites_por_objetivo)
        )
        for custo in custos
    ]


def agrupar_celulas(indices_grade, disponiveis):
    celulas = {}

    for indice in disponiveis:
        celulas.setdefault(indices_grade[indice], []).append(indice)

    return celulas


def escolher_lideres(arquivo, rng, n_grid=10, alpha=0.1, beta=4):
    """seleciona os lideres priorizando hipercubos menos ocupados"""
    indices_grade = criar_hipercubos(arquivo, n_grid, alpha)
    disponiveis = list(range(len(arquivo)))
    lideres = []

    for _ in range(3):
        if not disponiveis:
            lideres.append(lideres[-1])
            continue

        celulas = agrupar_celulas(indices_grade, disponiveis)
        grupos = list(celulas.values())

        pesos = np.array(
            [len(g) ** (-beta) for g in grupos],
            dtype=float
        )
        pesos /= pesos.sum()

        grupo = grupos[int(rng.choice(len(grupos), p=pesos))]
        indice = int(rng.choice(grupo))

        lideres.append(arquivo[indice])
        disponiveis.remove(indice)

    return lideres


def atualizar_arquivo(arquivo, candidato, capacidade, rng,
                      n_grid=10, alpha=0.1, gamma=2):
    """mantem solucoes nao dominadas e limita o arquivo por hipercubos"""
    chave = candidato['posicao'].tobytes()

    
    for sol in arquivo:
        # impede armazenar o mesmo subconjunto
        if sol['posicao'].tobytes() == chave:
            return arquivo

        # impede armazenar outro subconjunto com os mesmos objetivos
        if np.array_equal(sol['custo'], candidato['custo']):
            return arquivo

        # descarta o candidato se ele for dominado
        if domina(sol['custo'], candidato['custo']):
            return arquivo

    arquivo = [
        sol for sol in arquivo
        if not domina(candidato['custo'], sol['custo'])
    ]

    arquivo.append({
        'posicao': candidato['posicao'].copy(),
        'custo': candidato['custo'].copy()
    })

    while len(arquivo) > capacidade:
        indices_grade = criar_hipercubos(arquivo, n_grid, alpha)
        celulas = agrupar_celulas(indices_grade, range(len(arquivo)))
        grupos = list(celulas.values())

        pesos = np.array(
            [len(g) ** gamma for g in grupos],
            dtype=float
        )
        pesos /= pesos.sum()

        grupo = grupos[int(rng.choice(len(grupos), p=pesos))]
        arquivo.pop(int(rng.choice(grupo)))

    return arquivo



def selecionar_solucoes_apresentaveis(avaliacoes, arquivo_pareto,
                                      caminho='solucoes.csv',
                                      f1_min=0.992, fpr_max=0.002,
                                      quantidade_adicionais=5):
    """Pos-processamento reprodutivel, exclusivamente da VALIDACAO INTERNA.

    1) Preserva todas as mascaras do arquivo de Pareto de F1/FPR.
    2) Considera como alternativas apenas mascaras com F1 >= f1_min,
       FPR <= fpr_max e menos atributos que a menor mascara do Pareto.
    3) Elimina candidatas dominadas se tambem considerarmos, apenas
       para a triagem complementar, a quantidade de atributos.
    4) Seleciona uma candidata para cada uma das cinco menores
       quantidades distintas de atributos. Dentro da mesma quantidade,
       prefere maior F1, depois menor FPR, depois a primeira iteracao.

    Esta triagem NAO altera os objetivos, lideres ou dominancia do MOGWO.
    As alternativas NAO fazem parte do arquivo de Pareto F1/FPR.
    """
    dados = pd.DataFrame(avaliacoes).copy()
    if dados.empty or not arquivo_pareto:
        raise ValueError('Avaliacoes ou arquivo de Pareto estao vazios.')
    dados['mascara'] = dados['mascara'].astype(str)
    if dados['mascara'].duplicated().any():
        raise ValueError('avaliacoes contem mascaras duplicadas.')

    mascaras_pareto = {
        ''.join(map(str, np.asarray(s['posicao'], dtype=np.uint8)))
        for s in arquivo_pareto
    }
    pareto = dados[dados['mascara'].isin(mascaras_pareto)].copy()
    if len(pareto) != len(arquivo_pareto):
        raise ValueError('Nem todas as mascaras do pareto.pkl estao em avaliacoes.csv.')
    pareto.sort_values(['fpr', 'f1_score', 'n_atributos', 'mascara'],
                       ascending=[True, False, True, True], inplace=True)
    pareto['tipo'] = 'Pareto'
    pareto['solucao'] = [f'P{i:02d}' for i in range(1, len(pareto) + 1)]
    pareto['distancia_hamming_pareto'] = 0
    pareto['atributos_a_menos'] = 0

    minimo_pareto = int(pareto['n_atributos'].min())
    candidatas = dados.loc[
        (~dados['mascara'].isin(mascaras_pareto)) &
        (dados['f1_score'] >= f1_min) &
        (dados['fpr'] <= fpr_max) &
        (dados['n_atributos'] < minimo_pareto)
    ].copy()
    if candidatas.empty:
        print('Nenhuma alternativa compacta atende aos criterios; sera salvo apenas o Pareto.')
        compactas = candidatas
        quantas_eficientes = 0
    else:
        # Dominancia com 3 criterios so nesta triagem, NAO na otimizacao.
        criterios = np.column_stack([
            1.0 - candidatas['f1_score'].to_numpy(dtype=float),
            candidatas['fpr'].to_numpy(dtype=float),
            candidatas['n_atributos'].to_numpy(dtype=float)
        ])
        manter = np.ones(len(candidatas), dtype=bool)
        for i, candidato in enumerate(criterios):
            dominadores = (np.all(criterios <= candidato, axis=1) &
                           np.any(criterios < candidato, axis=1))
            manter[i] = not np.any(dominadores)
        compactas = candidatas.iloc[np.flatnonzero(manter)].copy()
        quantas_eficientes = len(compactas)
        compactas.sort_values(
            ['n_atributos', 'f1_score', 'fpr', 'iteracao', 'mascara'],
            ascending=[True, False, True, True, True],
            kind='stable', inplace=True
        )
        compactas.drop_duplicates(subset=['n_atributos'], keep='first', inplace=True)
        compactas = compactas.head(quantidade_adicionais).copy()
        compactas['tipo'] = 'Alternativa compacta'
        compactas['solucao'] = [f'A{i:02d}' for i in range(1, len(compactas) + 1)]
        mascaras_pareto_binarias = np.stack([
            np.asarray(s['posicao'], dtype=np.uint8) for s in arquivo_pareto
        ])
        compactas['distancia_hamming_pareto'] = [
            int(np.min(np.count_nonzero(
                mascaras_pareto_binarias !=
                np.fromiter((int(bit) for bit in mascara), dtype=np.uint8),
                axis=1
            )))
            for mascara in compactas['mascara']
        ]
        compactas['atributos_a_menos'] = minimo_pareto - compactas['n_atributos']

    selecionadas = pd.concat([pareto, compactas], ignore_index=True)
    selecionadas['etapa'] = 'validacao_interna'
    selecionadas = selecionadas[[
        'solucao', 'tipo', 'iteracao', 'mascara', 'n_atributos',
        'f1_score', 'fpr', 'distancia_hamming_pareto',
        'atributos_a_menos', 'etapa'
    ]]
    selecionadas.to_csv(caminho, index=False)
    print(f'Candidatas de boa qualidade com menos atributos: {len(candidatas)}; '
          f'apos triagem complementar: {quantas_eficientes}.')
    print(f'Solucoes salvas em {caminho}: {len(pareto)} Pareto e '
          f'{len(compactas)} alternativas compactas.')
    print(f'Triagem complementar: F1 >= {f1_min}, FPR <= {fpr_max}, '
          f'n_atributos < {minimo_pareto}.')
    print('Todas as metricas aqui sao de VALIDACAO INTERNA; o teste e separado.')
    return selecionadas


def bmogwo(X_treino, y_treino, k=3, n_lobos=30, max_iter=200,
           archive_size=100, caminho_save='pareto.pkl',
           seed=123, n_grid=10, alpha=0.1, beta=4, gamma=2,
           log_csv='historico.csv',
           avaliacoes_csv='avaliacoes.csv',
           solucoes_csv='solucoes.csv'):

    rng = np.random.default_rng(seed)
    y_treino = np.asarray(y_treino)

    indices = np.arange(len(y_treino))
    tr_idx, val_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y_treino,
        random_state=42
    )

    X_tr, X_val = X_treino[tr_idx], X_treino[val_idx]
    y_tr, y_val = y_treino[tr_idx], y_treino[val_idx]

    n_features = X_treino.shape[1]

    cache = {}
    arquivo = []
    populacao = []
    avaliacoes = []
    iteracao_atual = 0

    def criar_solucao(mascara):
        mascara = np.asarray(mascara, dtype=np.uint8).copy()

        if not mascara.any():
            mascara[int(rng.integers(n_features))] = 1

        chave = mascara.tobytes()

        if chave not in cache:
            custo = avaliar_subconjunto(
                mascara, X_tr, X_val, y_tr, y_val, k
            )

            cache[chave] = custo

            avaliacoes.append({
                'iteracao': iteracao_atual,
                'mascara': ''.join(map(str, mascara)),
                'n_atributos': int(mascara.sum()),
                'f1_score': 1.0 - custo[0],
                'fpr': custo[1]
            })

        return {
            'posicao': mascara,
            'custo': cache[chave].copy()
        }

    # Inicialização da população
    for i in range(n_lobos):
        densidade = (i + 1) / (n_lobos + 1)

        sol = criar_solucao(
            rng.random(n_features) < densidade
        )

        populacao.append(sol)

        arquivo = atualizar_arquivo(
            arquivo, sol, archive_size,
            rng, n_grid, alpha, gamma
        )

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)

        escritor.writerow([
            'iteracao',
            'tamanho_arquivo',
            'avaliacoes_unicas',
            'melhor_f1',
            'menor_fpr',
            'min_atributos',
            'lobos_distintos',
            'lideres_distintos_media',
            'diversidade_hamming',
            'novas_avaliacoes'
        ])

        inicio = time.perf_counter()

        for it in range(max_iter):
            iteracao_atual = it + 1
            avaliacoes_antes = len(cache)
            contagem_lideres = []

            a = 2 * (1 - it / max_iter)
            novos = []

            for lobo in populacao:

                # Seleção dos três líderes pelos hipercubos
                lideres = escolher_lideres(
                    arquivo, rng, n_grid, alpha, beta
                )

                contagem_lideres.append(
                    len({
                        lider['posicao'].tobytes()
                        for lider in lideres
                    })
                )

                # Movimentação original do GWO
                candidatos = []

                for lider in lideres:
                    A = a * (2 * rng.random(n_features) - 1)
                    C = 2 * rng.random(n_features)

                    D = np.abs(
                        C * lider['posicao'] - lobo['posicao']
                    )

                    candidatos.append(
                        lider['posicao'] - A * D
                    )

                media = np.mean(candidatos, axis=0)

                # Binarização probabilística
                prob = 1 / (
                    1 + np.exp(
                        -np.clip(10 * (media - 0.5), -50, 50)
                    )
                )

                mascara = (
                    rng.random(n_features) < prob
                ).astype(np.uint8)

                sol = criar_solucao(mascara)

                novos.append(sol)

                arquivo = atualizar_arquivo(
                    arquivo, sol, archive_size,
                    rng, n_grid, alpha, gamma
                )

            populacao = novos

            # Diagnósticos de diversidade
            mascaras = np.stack([
                sol['posicao'] for sol in populacao
            ])

            lobos_distintos = len(
                np.unique(mascaras, axis=0)
            )

            distancias = np.mean(
                mascaras[:, None, :] != mascaras[None, :, :],
                axis=2
            )

            diversidade_hamming = distancias[
                np.triu_indices(len(populacao), k=1)
            ].mean()

            lideres_distintos_media = np.mean(
                contagem_lideres
            )

            novas_avaliacoes = (
                len(cache) - avaliacoes_antes
            )

            custos = np.array([
                s['custo'] for s in arquivo
            ])

            quantidades = [
                int(s['posicao'].sum()) for s in arquivo
            ]

            escritor.writerow([
                it + 1,
                len(arquivo),
                len(cache),
                1 - custos[:, 0].min(),
                custos[:, 1].min(),
                min(quantidades),
                lobos_distintos,
                lideres_distintos_media,
                diversidade_hamming,
                novas_avaliacoes
            ])

            f.flush()

            tempo_decorrido = time.perf_counter() - inicio
            tempo_medio = tempo_decorrido / (it + 1)
            tempo_restante = (
                tempo_medio * (max_iter - (it + 1))
            )

            decorrido = str(timedelta(
                seconds=int(tempo_decorrido)
            ))

            restante = str(timedelta(
                seconds=int(tempo_restante)
            ))

            print(
                f"iteração {it+1}/{max_iter}  "
                f"pareto: {len(arquivo)}  "
                f"avaliados: {len(cache)}  "
                f"melhor f1: {1-custos[:, 0].min():.5f}  "
                f"menor fpr: {custos[:, 1].min():.5f}  ",
                flush=True
            )

            # Salvamento periódico
            if (it + 1) % 20 == 0:
                with open(caminho_save, 'wb') as saida:
                    pickle.dump(arquivo, saida)

                pd.DataFrame(avaliacoes).to_csv(
                    avaliacoes_csv,
                    index=False
                )

    # Salvamento final
    with open(caminho_save, 'wb') as saida:
        pickle.dump(arquivo, saida)

    pd.DataFrame(avaliacoes).to_csv(
        avaliacoes_csv,
        index=False
    )

    selecionar_solucoes_apresentaveis(
        avaliacoes, arquivo, caminho=solucoes_csv,
        f1_min=0.992, fpr_max=0.002, quantidade_adicionais=5
    )

    print(
        f"Concluído: {len(arquivo)} soluções não dominadas, "
        f"{len(cache)} máscaras avaliadas."
    )

    return arquivo


def testar_subconjuntos_aleatorios(X_amostra, y_amostra, k=3):
    import time

    rng = np.random.default_rng(2026)
    y_amostra = np.asarray(y_amostra)

    # Mesma divisao interna utilizada na otimizacao
    indices = np.arange(len(y_amostra))

    tr_idx, val_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y_amostra,
        random_state=42
    )

    X_tr = X_amostra[tr_idx]
    X_val = X_amostra[val_idx]
    y_tr = y_amostra[tr_idx]
    y_val = y_amostra[val_idx]

    n_features = X_amostra.shape[1]

    # Faixas de quantidade de atributos
    grupos = [
        (1, 5),
        (6, 10),
        (11, 20),
        (21, 35),
        (36, 55),
        (56, n_features)
    ]

    avaliacoes_por_grupo = 15
    resultados = []
    inicio = time.perf_counter()

    total = len(grupos) * avaliacoes_por_grupo
    contador = 0

    for minimo, maximo in grupos:
        for _ in range(avaliacoes_por_grupo):

            # Sorteia quantos atributos selecionar
            quantidade = int(rng.integers(minimo, maximo + 1))

            # Sorteia quais atributos serao selecionados
            mascara = np.zeros(n_features, dtype=np.uint8)
            selecionados = rng.choice(
                n_features, size=quantidade, replace=False
            )
            mascara[selecionados] = 1

            # Utiliza a mesma funcao de avaliacao do MOGWO
            custo = avaliar_subconjunto(
                mascara, X_tr, X_val, y_tr, y_val, k
            )

            resultados.append({
                'grupo': f'{minimo}-{maximo}',
                'n_atributos': quantidade,
                'mascara': ''.join(map(str, mascara)),
                'f1_score': 1.0 - custo[0],
                'fpr': custo[1]
            })

            contador += 1
            decorrido = time.perf_counter() - inicio
            restante = decorrido / contador * (total - contador)

            print(
                f'{contador}/{total} | '
                f'Atributos: {quantidade} | '
                f'F1: {1.0 - custo[0]:.6f} | '
                f'FPR: {custo[1]:.6f} | ',
                #f'Decorrido: {decorrido / 60:.1f} min | '
                #f'Restante estimado: {restante / 60:.1f} min',
                flush=True
            )

        # Salva os resultados apos cada grupo
        pd.DataFrame(resultados).to_csv(
            'teste_subconjuntos_aleatorios.csv', index=False
        )

    df_resultados = pd.DataFrame(resultados)

    print('\nResumo por grupo:')
    print(
        df_resultados.groupby('grupo', sort=False).agg(
            quantidade=('f1_score', 'count'),
            melhor_f1=('f1_score', 'max'),
            menor_fpr=('fpr', 'min')
        )
    )

    print('\nTeste concluido.')
    return df_resultados


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--selecionar':
        # Recria solucoes.csv com os arquivos da otimizacao JA CONCLUIDA.
        # Nao carrega o dataset, nao executa o MOGWO e nao usa teste.csv.
        with open('pareto.pkl', 'rb') as entrada:
            arquivo_anterior = pickle.load(entrada)
        dados_anterior = pd.read_csv('avaliacoes.csv', dtype={'mascara': str})
        selecionar_solucoes_apresentaveis(
            dados_anterior, arquivo_anterior, caminho='solucoes.csv',
            f1_min=0.992, fpr_max=0.002, quantidade_adicionais=5
        )
        sys.exit(0)
    elif len(sys.argv) > 1:
        raise SystemExit('Uso: python mogwo.py OU python mogwo.py --selecionar')

    CAMINHO_DATASET = 'data/MachineLearningCVE'

    # 1. Carregamento e pré-processamento
    df = preprocessar(carregar_dataset(CAMINHO_DATASET))

    # 2. Divisão externa: 70% treino e 30% teste
    X_train_scaled, X_test_scaled, y_train, y_test = dividir_normalizar(df)

    # 3. Amostra estratificada de 10% do treinamento
    X_amostra, y_amostra = criar_amostra(
        X_train_scaled,
        y_train,
        tamanho=0.10
    )

    # 4. MOGWO original: dois objetivos, SEM mutação, semente da busca 123.
    # Histórico, avaliações internas e Pareto são mantidos separadamente.
    arquivo_final = bmogwo(
        X_amostra,
        y_amostra.values,
        k=3,
        n_lobos=30,
        max_iter=200,
        archive_size=100,
        seed=123,
        caminho_save='pareto.pkl',
        log_csv='historico.csv',
        avaliacoes_csv='avaliacoes.csv',
        solucoes_csv='solucoes.csv'
    )
    print(f"soluções: {len(arquivo_final)}")
