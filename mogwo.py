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
# 4. FUNÇÕES DO BMOGWO
# ============================================================
def avaliar_subconjunto(posicao, X_treino, y_treino, k=3):
    features = np.where(posicao == 1)[0]
    if len(features) == 0:
        return np.array([1.0, 1.0])
    
    X_sub = X_treino[:, features]
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_treino, test_size=0.3, stratify=y_treino, random_state=None
    )
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_val)
    
    f1 = f1_score(y_val, y_pred, zero_division=0)
    TN = np.sum((y_pred == 0) & (y_val == 0))
    FP = np.sum((y_pred == 1) & (y_val == 0))
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    # Adiciona ruído pequeno para evitar que soluções diferentes
    # produzam valores idênticos de custo
    ruido = np.random.uniform(0, 0.001)
    
    return np.array([1 - f1 + ruido, fpr + ruido])

def domina(a, b):
    return np.all(a <= b) and np.any(a < b)

def determinar_dominancia(pop):
    n = len(pop)
    for i in range(n):
        pop[i]['dominado'] = False
    for i in range(n):
        for j in range(i):
            if not pop[j]['dominado']:
                if domina(pop[i]['custo'], pop[j]['custo']):
                    pop[j]['dominado'] = True
                elif domina(pop[j]['custo'], pop[i]['custo']):
                    pop[i]['dominado'] = True
                    break
    return pop

def get_nao_dominados(pop):
    return [p for p in pop if not p['dominado']]

def remover_duplicatas_arquivo(arquivo):
    vistos = set()
    unicos = []
    for p in arquivo:
        # Duplicata por custo, não por posição
        chave = (round(p['custo'][0], 6), round(p['custo'][1], 6))
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(p)
    return unicos

def criar_hipercubos(custos, n_grid, alpha):
    grade = []
    for j in range(custos.shape[0]):
        mn, mx = np.min(custos[j]), np.max(custos[j])
        dc = alpha * (mx - mn)
        limites = np.linspace(mn - dc, mx + dc, n_grid - 1)
        grade.append({'lower': np.concatenate([[-np.inf], limites]),
                      'upper': np.concatenate([limites, [np.inf]])})
    return grade

def get_grid_index(p, grade):
    sub = []
    for j, g in enumerate(grade):
        i = min(np.searchsorted(g['upper'], p['custo'][j], side='left'), len(g['upper']) - 1)
        sub.append(i)
    return sub[0] * len(grade[0]['upper']) + sub[1], tuple(sub)

def roleta(p):
    c = np.cumsum(p)
    r = np.random.rand()
    idx = np.where(r <= c)[0]
    return idx[0] if len(idx) > 0 else len(p) - 1

def get_celulas(arquivo):
    indices = [p['grid_index'] for p in arquivo]
    unicas = list(set(indices))
    contagens = [indices.count(c) for c in unicas]
    return unicas, contagens

def selecionar_lider(arquivo, beta=4):
    celulas, contagens = get_celulas(arquivo)
    p = np.array(contagens, dtype=float) ** (-beta)
    p /= p.sum()
    celula = celulas[roleta(p)]
    membros = [i for i, x in enumerate(arquivo) if x['grid_index'] == celula]
    return arquivo[np.random.choice(membros)]

def deletar_do_arquivo(arquivo, n, gamma=2):
    for _ in range(n):
        celulas, contagens = get_celulas(arquivo)
        p = np.array(contagens, dtype=float) ** gamma
        p /= p.sum()
        celula = celulas[roleta(p)]
        membros = [i for i, x in enumerate(arquivo) if x['grid_index'] == celula]
        arquivo.pop(np.random.choice(membros))
    return arquivo

def get_custos(pop):
    return np.array([p['custo'] for p in pop]).T

# ============================================================
# 5. LOOP PRINCIPAL DO BMOGWO
# ============================================================
def bmogwo(X_treino, y_treino, k=3, n_lobos=30, max_iter=200,
           archive_size=100, alpha=0.1, n_grid=10, beta=4, gamma=2,
           caminho_save='arquivo_pareto.pkl'):

    n_features = X_treino.shape[1]
    arquivo = []

    print("Inicializando população...")
    populacao = []
    for i in range(n_lobos):
        pos = np.random.randint(0, 2, size=n_features).astype(float)
        custo = avaliar_subconjunto(pos, X_treino, y_treino, k)
        populacao.append({'posicao': pos.copy(), 'custo': custo.copy(),
                          'dominado': False, 'grid_index': 0, 'grid_sub_index': (0,0)})
        print(f"  Lobo {i+1}/{n_lobos} | 1-F1={custo[0]:.4f} | FPR={custo[1]:.4f}")

    populacao = determinar_dominancia(populacao)
    arquivo = get_nao_dominados(populacao)
    grade = criar_hipercubos(get_custos(arquivo), n_grid, alpha)
    for p in arquivo:
        p['grid_index'], p['grid_sub_index'] = get_grid_index(p, grade)

    print(f"\nInicialização concluída. Arquivo: {len(arquivo)} soluções\n")

    for it in range(max_iter):
        a = 2 - it * (2 / max_iter)

        for i in range(n_lobos):
            Delta = selecionar_lider(arquivo, beta)
            rep2 = [p for p in arquivo if not np.array_equal(p['posicao'], Delta['posicao'])]
            Beta = selecionar_lider(rep2, beta) if rep2 else Delta
            rep3 = [p for p in rep2 if not np.array_equal(p['posicao'], Beta['posicao'])]
            Alpha = selecionar_lider(rep3, beta) if rep3 else Beta

            nova_pos = np.zeros(n_features)
            for lider in [Delta, Beta, Alpha]:
                c = 2 * np.random.rand(n_features)
                D = np.abs(c * lider['posicao'] - populacao[i]['posicao'])
                A = 2 * a * np.random.rand(n_features) - a
                nova_pos += lider['posicao'] - A * np.abs(D)
            nova_pos /= 3
            sig = 1 / (1 + np.exp(-10 * (nova_pos - 0.5)))
            nova_pos = (sig >= 0.5).astype(float)
            if np.sum(nova_pos) == 0:
                nova_pos[np.random.randint(n_features)] = 1

            populacao[i]['posicao'] = nova_pos
            populacao[i]['custo'] = avaliar_subconjunto(nova_pos, X_treino, y_treino, k)

        populacao = determinar_dominancia(populacao)
        arquivo.extend(get_nao_dominados(populacao))
        arquivo = determinar_dominancia(arquivo)
        arquivo = get_nao_dominados(arquivo)
        arquivo = remover_duplicatas_arquivo(arquivo)
        grade = criar_hipercubos(get_custos(arquivo), n_grid, alpha)
        for p in arquivo:
            p['grid_index'], p['grid_sub_index'] = get_grid_index(p, grade)

        if len(arquivo) > archive_size:
            arquivo = deletar_do_arquivo(arquivo, len(arquivo) - archive_size, gamma)
            grade = criar_hipercubos(get_custos(arquivo), n_grid, alpha)
            for p in arquivo:
                p['grid_index'], p['grid_sub_index'] = get_grid_index(p, grade)

        print(f"Iteração {it+1}/{max_iter} | Arquivo: {len(arquivo)} soluções")

        if (it + 1) % 10 == 0:
            with open(caminho_save, 'wb') as f:
                pickle.dump(arquivo, f)
            print(f"  --> Salvo (iteração {it+1})")

    with open(caminho_save, 'wb') as f:
        pickle.dump(arquivo, f)
    print(f"\nConcluído. {len(arquivo)} soluções salvas em {caminho_save}")
    return arquivo

# ============================================================
# 6. EXECUÇÃO
# ============================================================
if __name__ == '__main__':
    CAMINHO_DATASET = 'data/MachineLearningCVE'

    df = carregar_dataset(CAMINHO_DATASET)
    df = preprocessar(df)
    X_train_scaled, X_test_scaled, y_train, y_test = dividir_normalizar(df)
    X_amostra, y_amostra = criar_amostra(X_train_scaled, y_train, tamanho=0.05)
    k = escolher_k(X_amostra, y_amostra)

    arquivo_final = bmogwo(
    X_treino=X_amostra,
    y_treino=y_amostra.values,
    k=k,
    n_lobos=30,
    max_iter=200,
    archive_size=50,
    caminho_save='arquivo_pareto.pkl'
)