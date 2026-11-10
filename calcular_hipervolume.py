import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

PASTA = Path(__file__).resolve().parent

# Espaco de MINIMIZACAO: [1 - F1, FPR].
# Referencia principal ja usada nas analises anteriores.
# As referencias adicionais avaliam sensibilidade, sem mudar a selecao.
REFERENCIAS = {
    "principal": (0.01, 0.003),
    "sensibilidade_1": (0.02, 0.005),
    "sensibilidade_2": (0.05, 0.010),
}


def validar_pontos(pontos, nome):
    pontos = np.asarray(pontos, dtype=float).reshape(-1, 2)
    if pontos.shape[0] == 0 or not np.isfinite(pontos).all():
        raise ValueError(f"Pontos invalidos ou ausentes: {nome}")
    if np.any(pontos < 0) or np.any(pontos > 1):
        raise ValueError(f"Objetivos fora de [0,1]: {nome}")
    return pontos


def calcular_hv(pontos, referencia):
    """Calcula HV EXATO via biblioteca pymoo (2 objetivos)."""
    pontos = validar_pontos(pontos, "calculo de HV")
    referencia = np.asarray(referencia, dtype=float)
    # Valida a adequacao da referencia para TODOS os pontos fornecidos.
    if np.any(pontos >= referencia):
        raise ValueError(
            f"Ponto de referencia {referencia} nao e estritamente pior "
            "que todos os pontos nos dois objetivos. Reveja a referencia "
            "e mantenha-a igual entre as duas etapas."
        )
    return float(HV(ref_point=referencia).do(pontos))


def nao_dominados(pontos):
    """Mascara de pontos nao dominados; somente para relato diagnostico."""
    pontos = validar_pontos(pontos, "nao dominados")
    manter = np.ones(len(pontos), dtype=bool)
    for i, p in enumerate(pontos):
        for j, q in enumerate(pontos):
            if i != j and np.all(q <= p) and np.any(q < p):
                manter[i] = False
                break
    return manter


def ler_pareto(caminho):
    # Abra somente arquivos .pkl produzidos por voce: pickle nao e seguro
    # para conteudo de origem desconhecida.
    with caminho.open("rb") as arquivo:
        solucoes = pickle.load(arquivo)
    if not isinstance(solucoes, list) or not solucoes:
        raise ValueError("pareto.pkl nao contem uma lista valida.")
    mascaras, custos = [], []
    for i, sol in enumerate(solucoes):
        if not isinstance(sol, dict) or not {"posicao", "custo"} <= sol.keys():
            raise ValueError(f"Formato inesperado na entrada {i} de pareto.pkl")
        mascara = np.asarray(sol["posicao"], dtype=np.uint8)
        custo = np.asarray(sol["custo"], dtype=float)
        if mascara.ndim != 1 or not np.isin(mascara, [0, 1]).all():
            raise ValueError(f"Mascara invalida no Pareto, indice {i}")
        if custo.shape != (2,):
            raise ValueError("Esperados dois objetivos em pareto.pkl: [1-F1, FPR].")
        mascaras.append("".join(mascara.astype(str)))
        custos.append(custo)
    if len(set(mascaras)) != len(mascaras):
        raise ValueError("Mascaras repetidas no pareto.pkl")
    return mascaras, validar_pontos(custos, "validacao interna")


def evolucao_validacao(referencia, caminho, saida_csv, saida_png):
    if not caminho.exists():
        print(f"{caminho.name} ausente: a evolucao temporal sera ignorada.")
        return
    df = pd.read_csv(caminho)
    exigidas = {"iteracao", "f1_score", "fpr"}
    if not exigidas <= set(df.columns):
        raise ValueError(f"avaliacoes.csv precisa de: {sorted(exigidas)}")
    df = df.sort_values("iteracao", kind="stable")
    frente_acumulada = []
    registros = []
    for it, grupo in df.groupby("iteracao", sort=True):
        for f1, fpr in zip(grupo["f1_score"], grupo["fpr"]):
            ponto = np.array([1.0 - float(f1), float(fpr)])
            if np.any(ponto >= referencia):
                continue  # Nao contribui para a regiao delimitada pela referencia.
            if any(np.all(outro <= ponto) for outro in frente_acumulada):
                continue
            frente_acumulada = [
                outro for outro in frente_acumulada
                if not (np.all(ponto <= outro) and np.any(ponto < outro))
            ]
            frente_acumulada.append(ponto)
        hv = (calcular_hv(frente_acumulada, referencia)
              if frente_acumulada else 0.0)
        registros.append({
            "iteracao": int(it),
            "hipervolume_validacao": hv,
            "pontos_na_fronteira_acumulada": len(frente_acumulada),
        })
    hist = pd.DataFrame(registros)
    hist.to_csv(saida_csv, index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(hist["iteracao"], hist["hipervolume_validacao"], lw=2)
    ax.set(xlabel="Iteracao do MOGWO", ylabel="Hipervolume (validacao interna)",
           title="Evolucao do hipervolume durante a otimizacao")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(saida_png, dpi=300)
    plt.close(fig)
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    suf = f"_s{args.seed}"

    mascaras, validacao = ler_pareto(PASTA / f"pareto{suf}.pkl")
    print(f"semente {args.seed}: {len(mascaras)} solucoes no arquivo de pareto "
          f"({nao_dominados(validacao).sum()} nao dominadas na validacao)")

    registros = []
    for nome, referencia in REFERENCIAS.items():
        hv_validacao = calcular_hv(validacao, referencia)
        registros.append({
            "referencia": nome,
            "ref_erro_f1": referencia[0],
            "ref_fpr": referencia[1],
            "n_solucoes": len(mascaras),
            "hv_validacao": hv_validacao,
            "hv_validacao_normalizado": hv_validacao / (referencia[0] * referencia[1]),
        })
        print(f"referencia {nome} {referencia}: HV = {hv_validacao:.6e} "
              f"(normalizado {hv_validacao / (referencia[0] * referencia[1]):.3f})")

    pd.DataFrame(registros).to_csv(PASTA / f"hipervolume_resumo{suf}.csv", index=False)

    ref_principal = np.array(REFERENCIAS["principal"], dtype=float)
    evolucao = evolucao_validacao(
        ref_principal,
        PASTA / f"avaliacoes{suf}.csv",
        PASTA / f"hipervolume_evolucao{suf}.csv",
        PASTA / f"hipervolume_evolucao{suf}.png",
    )
    if evolucao is not None:
        hv_final_acumulado = float(evolucao["hipervolume_validacao"].iloc[-1])
        hv_final_pareto = calcular_hv(validacao, ref_principal)
        if not np.isclose(hv_final_acumulado, hv_final_pareto, rtol=1e-8, atol=1e-14):
            raise ValueError(
                "O HV acumulado das avaliacoes nao coincide com o HV do arquivo de Pareto. "
                "Verifique se os arquivos sao da mesma execucao."
            )

if __name__ == "__main__":
    main()