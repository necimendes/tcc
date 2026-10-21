"""LMuMOGWO experimental para selecao de atributos no CIC-IDS2017.

Reconstrucao independente inspirada em Al-Tashi et al. (2023):
https://doi.org/10.32604/csse.2023.039788

Fonte da arquitetura: MOGWO com arquivo de Pareto, hipercubos e lideres.
Equacoes do artigo implementadas: voo de Levy via Mantegna (24-28),
probabilidade decrescente de mutacao (29), e aceitacao baseada em dominancia
(Algoritmo 1). A funcao Mutate NAO e especificada no artigo; aqui consiste
na inversao de um bit sorteado. As equacoes GWO de movimentacao continuam
baseadas no codigo BMOGWO da pesquisa, e nao reproduzem literalmente todos
os passos intermediarios binarios (16-20) de Al-Tashi et al.

Assim, este e um LMuMOGWO adaptado/reconstruido, NAO o codigo oficial.
Objetivos deste TCC: minimizar [1-F1, FPR] com k-NN (em vez dos objetivos
numero de atributos/erro e classificador ANN utilizados no artigo).

O teste independente continua reservado: nao e utilizado para otimizar.
AVISO: preservamos o protocolo de normalizacao e a escolha de k do MOGWO
anterior por comparabilidade. A normalizacao anterior a validacao interna
e a escolha de k na amostra completa podem causar vazamento interno; nao
use estas metricas como teste final de generalizacao.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


ARTIGO = "https://doi.org/10.32604/csse.2023.039788"


# ---------- Protocolo experimental preservado do MOGWO do TCC ----------
def carregar_dataset(caminho: str) -> pd.DataFrame:
    dfs = []
    for nome in sorted(os.listdir(caminho)):
        if nome.lower().endswith(".csv"):
            parte = pd.read_csv(os.path.join(caminho, nome), low_memory=False)
            print(f"{nome}: {parte.shape}", flush=True)
            dfs.append(parte)
    if not dfs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {caminho!r}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset completo: {df.shape}", flush=True)
    return df


def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    if ' Label' not in df:
        raise KeyError("Coluna ' Label' do CIC-IDS2017 nao encontrada")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    antes = len(df)
    df.dropna(inplace=True)
    print(f"Removidos por ausente/infinito: {antes-len(df)}")
    antes = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removidos por duplicata: {antes-len(df)}")
    df[' Label'] = df[' Label'].map(
        lambda x: 0 if str(x).strip() == 'BENIGN' else 1
    )
    print(f"Total apos limpeza: {len(df)}")
    print(df[' Label'].value_counts())
    return df


def dividir_normalizar(df: pd.DataFrame, seed: int = 42):
    X = df.drop(columns=[' Label'])
    y = df[' Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Treino: {len(X_train_scaled)} | Teste reservado: {len(X_test_scaled)}")
    return X_train_scaled, X_test_scaled, y_train, y_test


def criar_amostra(X_train, y_train, tamanho: float = 0.10, seed: int = 42):
    amostrador = StratifiedShuffleSplit(
        n_splits=1, test_size=1-tamanho, random_state=seed
    )
    idx, _ = next(amostrador.split(X_train, y_train))
    y = np.asarray(y_train)[idx]
    print(f"Amostra estratificada: {len(idx)} registros")
    return X_train[idx], y


def escolher_k(X, y, candidatos=(3, 5, 7, 9)) -> int:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}
    for k in candidatos:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='f1', n_jobs=-1)
        resultados[k] = float(scores.mean())
        print(f"k={k} | F1 medio={scores.mean():.5f} | desvio={scores.std():.5f}")
    k_escolhido = max(resultados, key=resultados.get)
    print(f"k escolhido: {k_escolhido}")
    return k_escolhido


# ---------- Pareto, arquivo e hipercubos (estrutura do MOGWO) ----------
def domina(custo_a: np.ndarray, custo_b: np.ndarray) -> bool:
    return bool(np.all(custo_a <= custo_b) and np.any(custo_a < custo_b))


def criar_hipercubos(arquivo: list, n_grid: int = 10, alpha: float = 0.1):
    custos = np.asarray([sol['custo'] for sol in arquivo], dtype=float)
    limites = []
    for objetivo in range(custos.shape[1]):
        mn, mx = custos[:, objetivo].min(), custos[:, objetivo].max()
        margem = alpha * (mx-mn)
        limites.append(np.linspace(mn-margem, mx+margem, n_grid-1))
    return [
        tuple(int(np.searchsorted(g, custo[j], side='left'))
              for j, g in enumerate(limites))
        for custo in custos
    ]


def agrupar_celulas(ids_celula, indices):
    grupos = {}
    for i in indices:
        grupos.setdefault(ids_celula[i], []).append(i)
    return list(grupos.values())


def escolher_lideres(arquivo, rng, n_grid=10, alpha=0.1, beta=4):
    """Roleta entre hipercubos: probabilidade proporcional a ocupacao^-beta."""
    if not arquivo:
        raise ValueError("Arquivo vazio: nao e possivel selecionar lideres")
    celulas = criar_hipercubos(arquivo, n_grid, alpha)
    disponiveis = list(range(len(arquivo)))
    lideres = []
    for _ in range(3):
        if not disponiveis:
            lideres.append(lideres[-1])
            continue
        grupos = agrupar_celulas(celulas, disponiveis)
        pesos = np.asarray([len(g)**(-beta) for g in grupos], dtype=float)
        pesos /= pesos.sum()
        grupo = grupos[int(rng.choice(len(grupos), p=pesos))]
        i = int(rng.choice(grupo))
        lideres.append(arquivo[i])
        disponiveis.remove(i)
    return lideres


def atualizar_arquivo(arquivo, solucao, limite, rng,
                      n_grid=10, alpha=0.1, gamma=2):
    """Mantem um representante por par objetivo e somente nao dominadas."""
    chave = solucao['posicao'].tobytes()
    for antiga in arquivo:
        if antiga['posicao'].tobytes() == chave:
            return arquivo
        if np.array_equal(antiga['custo'], solucao['custo']):
            return arquivo
        if domina(antiga['custo'], solucao['custo']):
            return arquivo
    arquivo = [
        antiga for antiga in arquivo
        if not domina(solucao['custo'], antiga['custo'])
    ]
    arquivo.append({
        'posicao': solucao['posicao'].copy(),
        'custo': solucao['custo'].copy(),
    })
    while len(arquivo) > limite:
        celulas = criar_hipercubos(arquivo, n_grid, alpha)
        grupos = agrupar_celulas(celulas, range(len(arquivo)))
        pesos = np.asarray([len(g)**gamma for g in grupos], dtype=float)
        pesos /= pesos.sum()
        grupo = grupos[int(rng.choice(len(grupos), p=pesos))]
        arquivo.pop(int(rng.choice(grupo)))
    return arquivo


# ---------- Funcao wrapper k-NN ----------
def avaliar_subconjunto(mascara, X_tr, X_val, y_tr, y_val, k):
    features = np.flatnonzero(mascara)
    if len(features) == 0:
        return np.array([1., 1.])
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr[:, features], y_tr)
    pred = knn.predict(X_val[:, features])
    f1 = f1_score(y_val, pred, zero_division=0)
    tn, fp, _, _ = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    fpr = fp / (tn+fp) if tn+fp else 0.
    return np.array([1.-f1, fpr], dtype=float)


def prob_sigmoide(vetor):
    return 1. / (1. + np.exp(-np.clip(10.*(vetor-0.5), -50., 50.)))


def voo_levy_mantegna(rng, tamanho: int, indice_levy: float = 1.5):
    """Mantegna: S = u / abs(v)**(1/indice_levy), Eqs. 24-26 do artigo.

    Observacao: indice_levy=1.5 e escolha de reconstrucao: o artigo nao
    fixa explicitamente um unico valor para esse indice em sua tabela.
    """
    b = float(indice_levy)
    if not 0. < b < 2.:
        raise ValueError('indice_levy precisa pertencer a (0,2)')
    sig_u = (
        math.gamma(1. + b) * math.sin(math.pi*b/2.) / (
            math.gamma((1. + b)/2.) * b * 2.**((b-1.)/2.)
        )
    ) ** (1. / b)
    u = rng.normal(0., sig_u, size=tamanho)
    v = rng.normal(0., 1., size=tamanho)
    return u / (np.abs(v) + 1.e-12) ** (1. / b)


def probabilidade_mutacao(iteracao: int, max_iter: int, mu: float = 0.1):
    """Eq. 29: p_m = (1 - t/(T-1))**(1/mu), com t em 0..T-1."""
    if mu <= 0.:
        raise ValueError('mu precisa ser positivo')
    if max_iter <= 1:
        return 1.
    return float(np.clip((1. - iteracao/(max_iter-1.))**(1./mu), 0., 1.))


def mutar_um_bit(mascara, rng):
    """ESCOLHA DE IMPLEMENTACAO: o artigo nao especifica a funcao Mutate.

    Nesta reconstrucao, inverte-se um bit aleatorio em cada tentativa
    de mutacao. Nao e possivel afirmar que seja identico ao codigo oficial.
    """
    nova = mascara.copy()
    indice = int(rng.integers(0, len(nova)))
    nova[indice] = 1 - nova[indice]
    return nova


def media_hamming_normalizada(populacao):
    X = np.stack([p['posicao'] for p in populacao]).astype(bool)
    if len(X) < 2:
        return 0.
    matriz = np.mean(X[:, None, :] != X[None, :, :], axis=2)
    return float(matriz[np.triu_indices(len(X), 1)].mean())


def salvar_pkl(arquivo, caminho):
    with open(caminho, 'wb') as f:
        pickle.dump(arquivo, f)


def lmumogwo(X_amostra, y_amostra, *, k=3, n_lobos=30, max_iter=200,
              archive_size=100, seed=123, seed_validacao=42,
              n_grid=10, alpha=0.1, beta_grid=4, gamma_grid=2,
              indice_levy=1.5, escala_levy=0.01, mu=0.1,
              salvar_em='arquivo_pareto_lmumogwo.pkl',
              historico_em='historico_lmumogwo.csv',
              avaliacoes_em='avaliacoes_lmumogwo.csv',
              checkpoint_cada=20):
    """LMuMOGWO adaptado: retorno do arquivo de solucoes nao dominadas.

    Todos os candidatos usam uma mesma divisao interna, fixada por
    seed_validacao. O conjunto de teste externo nao participa da otimizacao.
    """
    if n_lobos < 1 or max_iter < 1 or archive_size < 1:
        raise ValueError('lobos, iteracoes e capacidade precisam ser positivos')
    rng = np.random.default_rng(seed)
    X_amostra = np.asarray(X_amostra)
    y_amostra = np.asarray(y_amostra)
    idx = np.arange(len(y_amostra))
    treino_idx, valid_idx = train_test_split(
        idx, test_size=0.30, stratify=y_amostra, random_state=seed_validacao
    )
    X_tr, X_val = X_amostra[treino_idx], X_amostra[valid_idx]
    y_tr, y_val = y_amostra[treino_idx], y_amostra[valid_idx]
    n_features = X_amostra.shape[1]
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
            custo = avaliar_subconjunto(mascara, X_tr, X_val, y_tr, y_val, k)
            cache[chave] = custo
            avaliacoes.append({
                'iteracao': iteracao_atual,
                'mascara': ''.join(map(str, mascara.tolist())),
                'n_atributos': int(mascara.sum()),
                'f1_score': float(1. - custo[0]),
                'fpr': float(custo[1]),
            })
        return {'posicao': mascara, 'custo': cache[chave].copy()}

    # Mesma diversidade de densidades iniciais do MOGWO do TCC.
    for i in range(n_lobos):
        densidade = (i + 1) / (n_lobos + 1)
        sol = criar_solucao(rng.random(n_features) < densidade)
        populacao.append(sol)
        arquivo = atualizar_arquivo(
            arquivo, sol, archive_size, rng, n_grid, alpha, gamma_grid
        )

    inicio = time.perf_counter()
    campos = [
        'iteracao', 'tamanho_arquivo', 'avaliacoes_unicas', 'melhor_f1',
        'menor_fpr', 'min_atributos', 'lobos_distintos',
        'diversidade_hamming', 'lideres_distintos', 'mutacoes_tentadas',
        'mutacoes_aceitas', 'p_mutacao', 'novas_avaliacoes', 'tempo_s',
    ]
    with open(historico_em, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for it in range(max_iter):
            iteracao_atual = it + 1
            anteriores = len(cache)
            a = 2. * (1. - it/max_iter)
            pm = probabilidade_mutacao(it, max_iter, mu)
            mutacoes_tentadas = 0
            mutacoes_aceitas = 0
            lideres = escolher_lideres(arquivo, rng, n_grid, alpha, beta_grid)
            lideres_distintos = len({l['posicao'].tobytes() for l in lideres})
            novos = []
            for lobo in populacao:
                # Movimento GWO preservado da versao atual da pesquisa.
                propostas = []
                for lider in lideres:
                    A = a*(2.*rng.random(n_features)-1.)
                    C = 2.*rng.random(n_features)
                    D = np.abs(C*lider['posicao'] - lobo['posicao'])
                    propostas.append(lider['posicao'] - A*D)
                media_gwo = np.mean(propostas, axis=0)

                # Artigo Eqs. 24-28: sigmoide + salto Levy (Mantegna).
                salto = escala_levy*voo_levy_mantegna(rng, n_features, indice_levy)
                prob = np.clip(prob_sigmoide(media_gwo) + salto, 0., 1.)
                mascara = (rng.random(n_features) < prob).astype(np.uint8)
                levy_sol = criar_solucao(mascara)
                escolhida = levy_sol

                # Artigo Eq. 29 + Alg. 1: mutacao com decisao de Pareto.
                # Funcao Mutate: inversao de um bit (suposicao documentada).
                if rng.random() < pm:
                    mutacoes_tentadas += 1
                    candidata = criar_solucao(mutar_um_bit(levy_sol['posicao'], rng))
                    if domina(candidata['custo'], levy_sol['custo']):
                        escolhida = candidata
                    elif not domina(levy_sol['custo'], candidata['custo']):
                        if rng.random() < 0.5:
                            escolhida = candidata
                    if escolhida is candidata:
                        mutacoes_aceitas += 1
                novos.append(escolhida)

            # Atualizacao do arquivo apos produzir a geracao (Algoritmo 2).
            for sol in novos:
                arquivo = atualizar_arquivo(
                    arquivo, sol, archive_size, rng, n_grid, alpha, gamma_grid
                )
            populacao = novos
            custos = np.asarray([sol['custo'] for sol in arquivo])
            quantidades = [int(sol['posicao'].sum()) for sol in arquivo]
            unicos = len({sol['posicao'].tobytes() for sol in populacao})
            decorrido = time.perf_counter()-inicio
            restante = (decorrido/(it+1))*(max_iter-it-1)
            escritor.writerow({
                'iteracao': it+1,
                'tamanho_arquivo': len(arquivo),
                'avaliacoes_unicas': len(cache),
                'melhor_f1': 1.-custos[:, 0].min(),
                'menor_fpr': custos[:, 1].min(),
                'min_atributos': min(quantidades),
                'lobos_distintos': unicos,
                'diversidade_hamming': media_hamming_normalizada(populacao),
                'lideres_distintos': lideres_distintos,
                'mutacoes_tentadas': mutacoes_tentadas,
                'mutacoes_aceitas': mutacoes_aceitas,
                'p_mutacao': pm,
                'novas_avaliacoes': len(cache)-anteriores,
                'tempo_s': round(decorrido, 3),
            })
            f.flush()
            print(
                f"Iteracao {it+1}/{max_iter} | Pareto: {len(arquivo)} | "
                f"mascaras: {len(cache)} | F1 max: {1.-custos[:, 0].min():.6f} | "
                f"FPR min: {custos[:, 1].min():.6f} | "
                f"Hamming: {media_hamming_normalizada(populacao):.4f} | "
                f"Decorrido: {timedelta(seconds=int(decorrido))} | "
                f"Restante est.: {timedelta(seconds=int(restante))}",
                flush=True,
            )
            if checkpoint_cada and (it+1) % checkpoint_cada == 0:
                salvar_pkl(arquivo, salvar_em)
                pd.DataFrame(avaliacoes).to_csv(avaliacoes_em, index=False)

    salvar_pkl(arquivo, salvar_em)
    pd.DataFrame(avaliacoes).to_csv(avaliacoes_em, index=False)
    print(f"Concluido: {len(arquivo)} pontos, {len(cache)} mascaras distintas.")
    return arquivo


def teste_rapido():
    """Teste funcional com dados artificiais, NAO resultado cientifico."""
    import tempfile
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=220, n_features=12, n_informative=6,
        n_redundant=3, random_state=12
    )
    X = MinMaxScaler().fit_transform(X)
    with tempfile.TemporaryDirectory() as pasta:
        resultado = lmumogwo(
            X, y, k=3, n_lobos=6, max_iter=5, archive_size=10,
            seed=123, seed_validacao=42, checkpoint_cada=0,
            salvar_em=os.path.join(pasta, 'teste.pkl'),
            historico_em=os.path.join(pasta, 'historico.csv'),
            avaliacoes_em=os.path.join(pasta, 'avaliacoes.csv'),
        )
    assert resultado
    assert len(resultado) <= 10
    assert len({s['custo'].tobytes() for s in resultado}) == len(resultado)
    for i, p in enumerate(resultado):
        for j, q in enumerate(resultado):
            if i != j:
                assert not domina(p['custo'], q['custo'])
    print('TESTE RAPIDO OK: arquivo nao dominado e sem custos duplicados')


def main():
    parser = argparse.ArgumentParser(description='LMuMOGWO experimental no CIC-IDS2017')
    parser.add_argument('--dataset', default='data/MachineLearningCVE')
    parser.add_argument('--lobos', type=int, default=30)
    parser.add_argument('--iteracoes', type=int, default=200)
    parser.add_argument('--arquivo-max', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--seed-validacao', type=int, default=42)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--escolher-k', action='store_true')
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--indice-levy', type=float, default=1.5)
    parser.add_argument('--escala-levy', type=float, default=0.01)
    parser.add_argument('--teste-rapido', action='store_true')
    args = parser.parse_args()
    if args.teste_rapido:
        teste_rapido()
        return

    df = preprocessar(carregar_dataset(args.dataset))
    X_train, _X_test, y_train, _y_test = dividir_normalizar(df, seed=42)
    X_amostra, y_amostra = criar_amostra(X_train, y_train, tamanho=0.10, seed=42)
    k = escolher_k(X_amostra, y_amostra) if args.escolher_k else args.k
    print(f"\nLMuMOGWO experimental | k={k} | lobos={args.lobos} | "
          f"iteracoes={args.iteracoes} | seed={args.seed} | "
          f"seed_validacao={args.seed_validacao}", flush=True)
    lmumogwo(
        X_amostra, y_amostra, k=k, n_lobos=args.lobos,
        max_iter=args.iteracoes, archive_size=args.arquivo_max,
        seed=args.seed, seed_validacao=args.seed_validacao,
        mu=args.mu, indice_levy=args.indice_levy,
        escala_levy=args.escala_levy,
    )


if __name__ == '__main__':
    main()
