"""Agrega os resultados de varias sementes do MOGWO com 3 objetivos.

Para cada semente le pareto_3obj_s<semente>.pkl, avaliacoes_3obj_s<semente>.csv e
solucoes_3obj_s<semente>.csv (VALIDACAO INTERNA) e gera:
  - resumo_sementes.csv       : uma linha por semente + media e desvio padrao
  - frequencia_atributos.csv  : frequencia de cada atributo nas solucoes selecionadas
Reaproveita as funcoes de calcular_hipervolume_3obj.py (mesmas referencias).

Uso:  python agregar_sementes.py --sementes 123 124 125 126 127
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import calcular_hipervolume_3obj as hv

PASTA = Path(__file__).resolve().parent
LIMIAR_F1_UTIL = 1.0 - hv.REFERENCIAS["principal"][0]  # F1 >= 0,985


def resumo_semente(s):
    suf = f"_3obj_s{s}"
    mascaras, custos = hv.ler_pareto(PASTA / f"pareto{suf}.pkl")
    f1 = 1.0 - custos[:, 0]
    natr = np.rint(custos[:, 2] * hv.N_ATRIBUTOS_TOTAL).astype(int)
    util = f1 >= LIMIAR_F1_UTIL
    ref_p = np.array(hv.REFERENCIAS["principal"], dtype=float)
    evol = hv.evolucao_validacao(PASTA / f"avaliacoes{suf}.csv", ref_p,
                                 PASTA / f"hipervolume_evolucao{suf}.csv",
                                 PASTA / f"hipervolume_evolucao{suf}.png")
    ganhos = evol.loc[evol["hipervolume_validacao"].diff() > 0, "iteracao"]
    linha = {
        "semente": s,
        "n_arquivo": len(mascaras),
        "n_uteis_f1": int(util.sum()),
        "n_dentro_ref_principal": hv.contar_dentro(custos, ref_p),
        "melhor_f1": f1.max(),
        "atributos_uteis_min": int(natr[util].min()) if util.any() else np.nan,
        "atributos_uteis_max": int(natr[util].max()) if util.any() else np.nan,
        "hv_frac_it100": float(evol.loc[evol["iteracao"] == 100, "fracao_do_hv_final"].iloc[0]),
        "ultimo_ganho_hv_iteracao": int(ganhos.iloc[-1]) if len(ganhos) else 0,
    }
    for nome, ref in hv.REFERENCIAS.items():
        linha[f"hv_norm_{nome}"] = hv.calcular_hv(custos, ref) / float(np.prod(ref))
    return linha


def frequencia_atributos(sementes):
    mats = []
    for s in sementes:
        sel = pd.read_csv(PASTA / f"solucoes_3obj_s{s}.csv", dtype={"mascara": str})
        mats.append(np.array([[int(c) for c in m] for m in sel["mascara"]]))
    todas = np.vstack(mats)
    df = pd.DataFrame({
        "atributo_n": np.arange(1, todas.shape[1] + 1),  # numeracao do Apendice
        "frequencia": todas.mean(axis=0),
        "n_solucoes": int(len(todas)),
    }).sort_values("frequencia", ascending=False)
    corr = {}
    for i, a in enumerate(sementes):
        for j, b in enumerate(sementes):
            if i < j:
                corr[(a, b)] = float(np.corrcoef(mats[i].mean(0), mats[j].mean(0))[0, 1])
    return df, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, nargs="+", required=True)
    args = ap.parse_args()

    linhas = [resumo_semente(s) for s in args.sementes]
    df = pd.DataFrame(linhas)
    num = df.drop(columns=["semente"])
    resumo = pd.concat([df, pd.DataFrame(
        [{"semente": "media", **num.mean().to_dict()},
         {"semente": "desvio_padrao", **num.std(ddof=1).to_dict()}])],
        ignore_index=True)
    resumo.to_csv(PASTA / "resumo_sementes.csv", index=False)
    print(resumo.round(4).to_string(index=False))

    freq, corr = frequencia_atributos(args.sementes)
    freq.to_csv(PASTA / "frequencia_atributos.csv", index=False)
    print("\nAtributos mais frequentes (numeracao do Apendice):")
    print(freq.head(10).round(3).to_string(index=False))
    print("\nAtributos em TODAS as solucoes selecionadas:",
          freq.loc[freq["frequencia"] == 1.0, "atributo_n"].tolist())
    print("Correlacao das frequencias entre sementes:",
          {k: round(v, 2) for k, v in corr.items()})
    print("\nOBS.: desvio padrao amostral (n-1); metricas de VALIDACAO INTERNA.")


if __name__ == "__main__":
    main()
