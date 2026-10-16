import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pickle

# ============================================================
# 1. CARREGAMENTO DO DATASET
# ============================================================
def carregar_dataset(caminho):
    dfs = []
    for arquivo in sorted(os.listdir(caminho)):
        if arquivo.endswith('.csv'):
            df_temp = pd.read_csv(os.path.join(caminho, arquivo), low_memory=False)
            print(f"{arquivo}: {df_temp.shape}")
            dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nDataset completo: {df.shape}")
    return df

# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    antes = len(df)
    df.dropna(inplace=True)
    print(f"Removidos por ausente/infinito: {antes - len(df)}")
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removidos por duplicata: {antes - len(df)}")
    df[' Label'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)
    print(f"Total após limpeza: {len(df)}")
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
    print(f"Treino: {X_train_scaled.shape[0]} | Teste: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test

def criar_amostra(X_train_scaled, y_train, tamanho=0.10):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-tamanho, random_state=42)
    idx, _ = next(sss.split(X_train_scaled, y_train))
    X_amostra = X_train_scaled[idx]
    y_amostra = y_train.iloc[idx].reset_index(drop=True)
    print(f"Amostra: {len(X_amostra)} registros")
    return X_amostra, y_amostra

# ============================================================
# 3. VALIDAÇÃO CRUZADA PARA ESCOLHA DO K
# ============================================================
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

# ============================================================
# 4. AVALIAÇÃO REPRODUTÍVEL E ARQUIVO NÃO DOMINADO
# ============================================================
from sklearn.metrics import confusion_matrix
import csv
import copy


def domina(a, b):
    """Ambos os custos são minimizados: [1-F1, FPR]."""
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
    return np.array([1.0 - f1, fp / (fp + tn) if fp + tn else 0.0])


def atualizar_arquivo(arquivo, candidato, capacidade, rng):
    """Mantém apenas máscaras distintas não dominadas, sem compartilhar referências."""
    chave = tuple(candidato['posicao'].astype(np.uint8))
    for sol in arquivo:
        if tuple(sol['posicao'].astype(np.uint8)) == chave:
            return arquivo
        if domina(sol['custo'], candidato['custo']):
            return arquivo
    arquivo = [s for s in arquivo if not domina(candidato['custo'], s['custo'])]
    arquivo.append({'posicao': candidato['posicao'].copy(),
                    'custo': candidato['custo'].copy()})
    if len(arquivo) > capacidade:
        # Crowding distance em espaço normalizado; conserva extremos.
        custos = np.array([s['custo'] for s in arquivo])
        dist = np.zeros(len(arquivo))
        for j in range(2):
            ordem = np.argsort(custos[:, j], kind='stable')
            dist[ordem[0]] = dist[ordem[-1]] = np.inf
            amplitude = custos[ordem[-1], j] - custos[ordem[0], j]
            if amplitude > 0:
                for z in range(1, len(ordem) - 1):
                    dist[ordem[z]] += (custos[ordem[z+1], j] - custos[ordem[z-1], j]) / amplitude
        finitos = np.flatnonzero(np.isfinite(dist))
        remover = int(rng.choice(finitos[np.isclose(dist[finitos], np.min(dist[finitos]))])) if len(finitos) else int(rng.integers(len(arquivo)))
        arquivo.pop(remover)
    return arquivo


def escolher_lideres(arquivo, rng):
    """Preferência por soluções menos congestionadas, sem exigir três distintas."""
    custos = np.array([s['custo'] for s in arquivo])
    n = len(arquivo)
    if n == 1:
        return [arquivo[0]] * 3
    normal = (custos - custos.min(axis=0)) / np.maximum(np.ptp(custos, axis=0), 1e-12)
    dist = np.linalg.norm(normal[:, None, :] - normal[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    diversidade = np.min(dist, axis=1) + 0.05
    probs = diversidade / diversidade.sum()
    indices = rng.choice(n, size=3, replace=n < 3, p=probs)
    return [arquivo[int(i)] for i in indices]


def bmogwo(X_treino, y_treino, k=3, n_lobos=30, max_iter=200,
           archive_size=100, caminho_save='arquivo_pareto.pkl',
           seed=42, taxa_mutacao=0.03, fracao_aleatoria=0.10,
           log_csv='historico_mogwo.csv', seed_validacao=42):
    """MOGWO binário com avaliação fixa e mutação para diversidade."""
    rng = np.random.default_rng(seed)
    y_treino = np.asarray(y_treino)
    # A mesma divisão é usada em TODAS as avaliações dos lobos.
    indices = np.arange(len(y_treino))
    tr_idx, val_idx = train_test_split(indices, test_size=0.30,
                                       stratify=y_treino, random_state=seed_validacao)
    X_tr, X_val = X_treino[tr_idx], X_treino[val_idx]
    y_tr, y_val = y_treino[tr_idx], y_treino[val_idx]
    n_features = X_treino.shape[1]
    cache = {}
    arquivo = []
    populacao = []

    def criar_solucao(mascara):
        mascara = np.asarray(mascara, dtype=np.uint8).copy()
        if not mascara.any():
            mascara[int(rng.integers(n_features))] = 1
        chave = mascara.tobytes()
        if chave not in cache:
            cache[chave] = avaliar_subconjunto(mascara, X_tr, X_val, y_tr, y_val, k)
        return {'posicao': mascara, 'custo': cache[chave].copy()}

    for i in range(n_lobos):
        densidade = (i + 1) / (n_lobos + 1)
        sol = criar_solucao(rng.random(n_features) < densidade)
        populacao.append(sol)
        arquivo = atualizar_arquivo(arquivo, sol, archive_size, rng)

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        escritor.writerow(['iteracao', 'tamanho_arquivo', 'avaliacoes_unicas',
                           'melhor_f1', 'menor_fpr', 'min_atributos'])
        for it in range(max_iter):
            a = 2 * (1 - it / max_iter)
            novos = []
            for lobo in populacao:
                lideres = escolher_lideres(arquivo, rng)
                candidatos = []
                for lider in lideres:
                    A = a * (2 * rng.random(n_features) - 1)
                    C = 2 * rng.random(n_features)
                    D = np.abs(C * lider['posicao'] - lobo['posicao'])
                    candidatos.append(lider['posicao'] - A * D)
                media = np.mean(candidatos, axis=0)
                # Probabilidade sigmoide; não limiar determinístico.
                prob = 1 / (1 + np.exp(-np.clip(10 * (media - 0.5), -50, 50)))
                mascara = (rng.random(n_features) < prob).astype(np.uint8)
                flips = rng.random(n_features) < taxa_mutacao
                mascara[flips] = 1 - mascara[flips]
                if rng.random() < fracao_aleatoria:
                    densidade = rng.uniform(1 / n_features, 0.7)
                    mascara = (rng.random(n_features) < densidade).astype(np.uint8)
                sol = criar_solucao(mascara)
                novos.append(sol)
                arquivo = atualizar_arquivo(arquivo, sol, archive_size, rng)
            populacao = novos
            custos = np.array([s['custo'] for s in arquivo])
            quantidades = [int(s['posicao'].sum()) for s in arquivo]
            escritor.writerow([it + 1, len(arquivo), len(cache),
                               1 - custos[:, 0].min(), custos[:, 1].min(), min(quantidades)])
            f.flush()
            print(f"Iteração {it+1}/{max_iter} | Pareto: {len(arquivo)} | "
                  f"avaliados: {len(cache)} | melhor F1: {1-custos[:,0].min():.5f} | "
                  f"menor FPR: {custos[:,1].min():.5f}")
            if (it + 1) % 20 == 0:
                with open(caminho_save, 'wb') as saida:
                    pickle.dump(arquivo, saida)
    with open(caminho_save, 'wb') as saida:
        pickle.dump(arquivo, saida)
    print(f"Concluído: {len(arquivo)} soluções não dominadas, {len(cache)} máscaras avaliadas.")
    return arquivo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Repetir o MOGWO com outra semente, preservando a validação.')
    parser.add_argument('--seed', type=int, default=123, help='Semente da busca (padrão: 123)')
    parser.add_argument('--iteracoes', type=int, default=200, help='Número de iterações (padrão: 200)')
    parser.add_argument('--dados', default='data/MachineLearningCVE', help='Pasta com arquivos CSV')
    args = parser.parse_args()

    if args.seed == 42:
        print('AVISO: a semente 42 repete a primeira execução; use 123 para comparar.')
    if args.iteracoes != 200:
        print('AVISO: para comparar a replicação com a execução original, prefira 200 iterações.')

    # Mantemos constante k=3, obtido na validação cruzada da primeira execução.
    # Repetir escolher_k() não traz uma nova comparação do otimizador e custa tempo.
    k = 3
    CAMINHO_DATASET = args.dados
    df = preprocessar(carregar_dataset(CAMINHO_DATASET))
    X_train_scaled, _X_test_scaled, y_train, _y_test = dividir_normalizar(df)
    X_amostra, y_amostra = criar_amostra(X_train_scaled, y_train, tamanho=0.10)
    print(f'\nREPLICAÇÃO | semente da busca: {args.seed} | semente da validação: 42 | k={k}')

    sufixo = f'seed{args.seed}'
    bmogwo(X_amostra, y_amostra.values, k=k,
           n_lobos=30, max_iter=args.iteracoes, archive_size=100,
           caminho_save=f'arquivo_pareto_{sufixo}.pkl', seed=args.seed,
           taxa_mutacao=0.03, fracao_aleatoria=0.10,
           log_csv=f'historico_{sufixo}.csv', seed_validacao=42)
