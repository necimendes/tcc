"""Hipervolume (HV) para o experimento com 3 objetivos.

Espaco de MINIMIZACAO: [1 - F1, FPR, n_atributos / 78].

O que o script faz:
  1) HV do arquivo de Pareto completo na VALIDACAO INTERNA (pareto_3obj_s<semente>.pkl).
  2) HV das solucoes selecionadas (as que foram ao teste) na validacao e no TESTE,
     ambos descritivos e NAO comparaveis entre si (modelos treinados com
     quantidades de dados diferentes).
  3) Evolucao do HV por iteracao, reconstruida de avaliacoes_3obj_s<semente>.csv,
     com verificacao de que o ultimo valor coincide com o do arquivo .pkl.

Pontos fora da caixa definida pelo ponto de referencia (por exemplo, subconjuntos
degenerados com F1 proximo de 0) NAO contribuem ao HV; o script os ignora e informa
quantos foram ignorados. O ponto de referencia e o criterio de utilidade, portanto
devem ser descritos juntos na metodologia.

Uso:  python calcular_hipervolume_3obj.py --seed 123
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
N_ATRIBUTOS_TOTAL = 78

# Referencia PRINCIPAL = criterio de utilidade: F1 >= 0,985, FPR <= 0,003 e no
# maximo 50% dos atributos (39). As demais avaliam sensibilidade.
REFERENCIAS = {
    "principal": (0.015, 0.003, 0.5),
    "sensibilidade_1": (0.02, 0.005, 0.5),
    "sensibilidade_2": (0.05, 0.010, 0.5),
}

try:  # HV exato do pymoo, se instalado
    from pymoo.indicators.hv import HV as _HV_PYMOO
except Exception:  # pragma: no cover
    _HV_PYMOO = None


# ----------------------------------------------------------------------------
# Hipervolume
# ----------------------------------------------------------------------------
def _hv2d(pontos, ref):
    """HV exato em 2D (minimizacao)."""
    pontos = pontos[np.argsort(pontos[:, 0], kind="stable")]
    volume, melhor_y = 0.0, ref[1]
    for x, y in pontos:
        if y < melhor_y:
            volume += (ref[0] - x) * (melhor_y - y)
            melhor_y = y
    return volume


def hv_interno(pontos, ref):
    """HV exato em 3D por fatiamento no terceiro objetivo."""
    if len(pontos) == 0:
        return 0.0
    pontos = pontos[np.argsort(pontos[:, 2], kind="stable")]
    volume = 0.0
    for i in range(len(pontos)):
        z_prox = pontos[i + 1, 2] if i + 1 < len(pontos) else ref[2]
        volume += (z_prox - pontos[i, 2]) * _hv2d(pontos[: i + 1, :2], ref[:2])
    return float(volume)


def calcular_hv(pontos, ref):
    """HV dos pontos DENTRO da caixa da referencia (os demais sao ignorados)."""
    pontos = np.asarray(pontos, dtype=float).reshape(-1, 3)
    ref = np.asarray(ref, dtype=float)
    dentro = pontos[np.all(pontos < ref, axis=1)]
    if len(dentro) == 0:
        return 0.0
    hv = hv_interno(dentro, ref)
    if _HV_PYMOO is not None:
        hv_pymoo = float(_HV_PYMOO(ref_point=ref).do(dentro))
        if not np.isclose(hv, hv_pymoo, rtol=1e-8, atol=1e-14):
            raise ValueError(f"HV interno ({hv}) difere do pymoo ({hv_pymoo}).")
        return hv_pymoo
    return hv


def contar_dentro(pontos, ref):
    pontos = np.asarray(pontos, dtype=float).reshape(-1, 3)
    return int(np.all(pontos < np.asarray(ref, dtype=float), axis=1).sum())


def nao_dominados(pontos):
    pontos = np.asarray(pontos, dtype=float).reshape(-1, 3)
    return np.array([
        not np.any(np.all(pontos <= p, axis=1) & np.any(pontos < p, axis=1))
        for p in pontos
    ])


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def ler_pareto(caminho):
    # pickle so deve ser aberto se produzido por voce.
    with caminho.open("rb") as arquivo:
        solucoes = pickle.load(arquivo)
    if not isinstance(solucoes, list) or not solucoes:
        raise ValueError(f"{caminho.name} nao contem uma lista valida.")
    mascaras, custos = [], []
    for i, sol in enumerate(solucoes):
        mascara = np.asarray(sol["posicao"], dtype=np.uint8)
        custo = np.asarray(sol["custo"], dtype=float)
        if custo.shape != (3,):
            raise ValueError(
                f"Entrada {i}: esperados 3 objetivos [1-F1, FPR, n/78]; "
                f"veio {custo.shape}. Este e o .pkl da rodada de 2 objetivos?")
        mascaras.append("".join(mascara.astype(str)))
        custos.append(custo)
    if len(set(mascaras)) != len(mascaras):
        raise ValueError("Mascaras repetidas no pareto.")
    return mascaras, np.vstack(custos)


def ler_teste(caminho, mascaras_pareto, custos_pareto):
    """Pontos de validacao e de teste das solucoes avaliadas externamente."""
    teste = pd.read_csv(caminho, dtype={"mascara": str}, encoding="utf-8-sig")
    exigidas = {"solucao", "mascara", "n_atributos", "f1_teste", "fpr_teste"}
    if not exigidas <= set(teste.columns):
        raise ValueError(f"{caminho.name} precisa das colunas {sorted(exigidas)}")
    if teste["mascara"].duplicated().any():
        raise ValueError("Mascaras duplicadas no arquivo de teste.")
    indice = {m: i for i, m in enumerate(mascaras_pareto)}
    faltantes = [m for m in teste["mascara"] if m not in indice]
    if faltantes:
        raise ValueError(
            f"{len(faltantes)} mascara(s) do teste nao estao no pareto. "
            "Verifique se sao da mesma execucao (mesma semente).")
    validacao = custos_pareto[[indice[m] for m in teste["mascara"]]]
    pontos_teste = np.column_stack([
        1.0 - pd.to_numeric(teste["f1_teste"]).to_numpy(float),
        pd.to_numeric(teste["fpr_teste"]).to_numpy(float),
        pd.to_numeric(teste["n_atributos"]).to_numpy(float) / N_ATRIBUTOS_TOTAL,
    ])
    return teste["solucao"].tolist(), validacao, pontos_teste


# ----------------------------------------------------------------------------
# Evolucao do HV
# ----------------------------------------------------------------------------
def evolucao_validacao(caminho, ref, saida_csv, saida_png):
    df = pd.read_csv(caminho)
    exigidas = {"iteracao", "f1_score", "fpr", "n_atributos"}
    if not exigidas <= set(df.columns):
        raise ValueError(f"{caminho.name} precisa de {sorted(exigidas)}")
    ref = np.asarray(ref, dtype=float)
    df = df.sort_values("iteracao", kind="stable")
    frente = np.empty((0, 3))
    registros = []
    for it, grupo in df.groupby("iteracao", sort=True):
        pts = np.column_stack([
            1.0 - grupo["f1_score"].to_numpy(float),
            grupo["fpr"].to_numpy(float),
            grupo["n_atributos"].to_numpy(float) / N_ATRIBUTOS_TOTAL,
        ])
        for p in pts:
            if np.any(p >= ref):
                continue
            if len(frente) and np.any(np.all(frente <= p, axis=1)):
                continue
            if len(frente):
                frente = frente[~np.all(p <= frente, axis=1)]
            frente = np.vstack([frente, p])
        registros.append({
            "iteracao": int(it),
            "hipervolume_validacao": calcular_hv(frente, ref),
            "pontos_na_fronteira_acumulada": len(frente),
        })
    hist = pd.DataFrame(registros)
    caixa = float(np.prod(ref))
    final = hist["hipervolume_validacao"].iloc[-1]
    hist["hv_normalizado"] = hist["hipervolume_validacao"] / caixa
    hist["fracao_do_hv_final"] = hist["hipervolume_validacao"] / final if final else np.nan
    hist.to_csv(saida_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(hist["iteracao"], hist["hipervolume_validacao"], lw=2)
    ax.set(xlabel="Iteracao do MOGWO", ylabel="Hipervolume (validacao interna)",
           title="Evolucao do hipervolume (3 objetivos)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(saida_png, dpi=300)
    plt.close(fig)
    return hist


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    suf = f"_3obj_s{args.seed}"

    mascaras, custos = ler_pareto(PASTA / f"pareto{suf}.pkl")
    caminho_teste = PASTA / f"teste{suf}.csv"

    if caminho_teste.exists():
        ids, val_sel, teste_sel = ler_teste(caminho_teste, mascaras, custos)
        print(f"Arquivo de Pareto: {len(mascaras)} solucoes | avaliadas no teste: {len(ids)}")
        print(
            f"Nao dominadas (3D) entre as avaliadas - "
            f"validacao: {nao_dominados(val_sel).sum()} | "
            f"teste: {nao_dominados(teste_sel).sum()}"
        )
    else:
        ids = []
        val_sel = None
        teste_sel = None
        print(f"Arquivo de Pareto: {len(mascaras)} solucoes")
        print("Arquivo de teste nao encontrado. Avaliacao externa ignorada.")

    # Os calculos da fronteira completa ocorrem com ou sem arquivo de teste.
    registros = []
    for nome, ref in REFERENCIAS.items():
        caixa = float(np.prod(ref))
        hv_arq = calcular_hv(custos, ref)
        hv_val = calcular_hv(val_sel, ref) if val_sel is not None else np.nan
        hv_tst = calcular_hv(teste_sel, ref) if teste_sel is not None else np.nan

        registros.append({
            "referencia": nome,
            "ref_erro_f1": ref[0],
            "ref_fpr": ref[1],
            "ref_frac_atributos": ref[2],
            "n_pareto_total": len(mascaras),
            "n_pareto_dentro_da_referencia": contar_dentro(custos, ref),
            "hv_arquivo_validacao": hv_arq,
            "hv_arquivo_validacao_normalizado": hv_arq / caixa,
            "n_avaliadas_teste": len(ids),
            "hv_selecionadas_validacao": hv_val,
            "hv_selecionadas_teste": hv_tst,
            "hv_selecionadas_teste_normalizado": hv_tst / caixa,
        })
        print(f"\nReferencia {nome}: [1-F1={ref[0]}, FPR={ref[1]}, n/78={ref[2]}]")
        print(f"  Pareto dentro da referencia: {contar_dentro(custos, ref)}/{len(mascaras)}")
        print(f"  HV arquivo (validacao):          {hv_arq:.6e}  (normalizado {hv_arq / caixa:.4f})")
        if val_sel is not None:
            print(f"  HV selecionadas (validacao):     {hv_val:.6e}")
            print(f"  HV selecionadas (teste externo): {hv_tst:.6e}  (normalizado {hv_tst / caixa:.4f})")

    pd.DataFrame(registros).to_csv(PASTA / f"hipervolume_resumo{suf}.csv", index=False)
    print(f"\nGerado: hipervolume_resumo{suf}.csv")

    ref_p = np.array(REFERENCIAS["principal"], dtype=float)
    hist = evolucao_validacao(
        PASTA / f"avaliacoes{suf}.csv", ref_p,
        PASTA / f"hipervolume_evolucao{suf}.csv",
        PASTA / f"hipervolume_evolucao{suf}.png",
    )
    hv_pkl = calcular_hv(custos, ref_p)
    hv_acum = float(hist["hipervolume_validacao"].iloc[-1])
    if not np.isclose(hv_acum, hv_pkl, rtol=1e-8, atol=1e-14):
        raise ValueError(
            "HV acumulado das avaliacoes nao coincide com o do .pkl. "
            "Verifique se os arquivos sao da mesma execucao."
        )
    print("Verificacao: HV acumulado das avaliacoes == HV do arquivo de Pareto.")
    its = hist["iteracao"].tolist()
    marcos = [i for i in (1, 10, 50, 100, 150, 175, 190) if i in its]
    print("Fracao do HV final (ref. principal): " + ", ".join(
        f"it {i}: {hist.loc[hist['iteracao'] == i, 'fracao_do_hv_final'].iloc[0]:.3f}"
        for i in marcos
    ))
    if teste_sel is not None:
        print("\nATENCAO: HV de validacao e de teste sao descritivos e NAO comparaveis")
        print("entre si (modelos treinados com quantidades de dados diferentes).")


if __name__ == "__main__":
    main()
