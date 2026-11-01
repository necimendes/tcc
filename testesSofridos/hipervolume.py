from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent
ARQUIVO_SOLUCOES = PASTA / "solucoes.csv"
ARQUIVO_AVALIACOES = PASTA / "avaliacoes.csv"

REF_F1 = 0.990
REF_FPR = 0.003
REFERENCIA = np.array([1.0 - REF_F1, REF_FPR], dtype=float)


def filtrar_pontos_na_referencia(df, ref=REFERENCIA):
    custos = np.column_stack([
        1.0 - df["f1_score"].to_numpy(dtype=float),
        df["fpr"].to_numpy(dtype=float),
    ])
    return custos[np.all(custos <= ref, axis=1)]


def frente_nao_dominada(custos):
    frente = []
    for ponto in custos:
        if any(np.all(outro <= ponto) for outro in frente):
            continue
        frente = [outro for outro in frente
                  if not (np.all(ponto <= outro) and np.any(ponto < outro))]
        frente.append(np.asarray(ponto, dtype=float))
    return np.array(frente, dtype=float).reshape(-1, 2)


def hipervolume_2d(custos, ref=REFERENCIA):
    if len(custos) == 0:
        return 0.0
    dentro = np.asarray(custos, dtype=float)
    dentro = dentro[np.all(dentro <= ref, axis=1)]
    if not len(dentro):
        return 0.0
    dentro = frente_nao_dominada(dentro)
    dentro = dentro[np.argsort(dentro[:, 0])]
    resultado = 0.0
    limite_fpr = ref[1]
    for erro_f1, fpr in dentro:
        if fpr < limite_fpr:
            resultado += (ref[0] - erro_f1) * (limite_fpr - fpr)
            limite_fpr = fpr
    return float(resultado)


def main():
    solucoes = pd.read_csv(ARQUIVO_SOLUCOES)
    avaliacoes = pd.read_csv(ARQUIVO_AVALIACOES)
    pareto = solucoes[solucoes["tipo"].astype(str).str.strip().str.lower() == "pareto"].copy()
    if pareto.empty:
        raise ValueError("cade pareto em solucoes.csv")
    if not {"f1_score", "fpr"}.issubset(solucoes.columns):
        raise ValueError("solucoes.csv precisa ter as colunas f1_score e fpr")

    pontos = filtrar_pontos_na_referencia(pareto)
    if len(pontos) != len(pareto):
        raise ValueError("o ponto de referencia deve ser pior que todas as solucoes de pareto")
    frente = frente_nao_dominada(pontos)
    if len(frente) != len(pareto):
        raise ValueError("as solucoes rotuladas como pareto nao sao todas nao dominadas entre si")

    hv_pareto = hipervolume_2d(pontos)
    hv_30 = hipervolume_2d(filtrar_pontos_na_referencia(solucoes))
    area_referencia = REFERENCIA[0] * REFERENCIA[1]

    print(f"solucoes na fronteira: {len(pareto)}")
    print(f"ponto de referencia: (1-F1={REFERENCIA[0]:.6f}, FPR={REFERENCIA[1]:.6f})")
    print(f"hv da fronteira: {hv_pareto:.12e}")
    print(f"hv das {len(solucoes)} solucoes reunidas: {hv_30:.12e}")
    print(f"area ocupada dentro do retangulo de referencia: {100*hv_pareto/area_referencia:.3f}%")

    avaliacoes = avaliacoes.sort_values("iteracao", kind="stable")
    candidatos = []
    registros = []
    for iteracao, grupo in avaliacoes.groupby("iteracao", sort=True):
        grupo_custos = filtrar_pontos_na_referencia(grupo)
        for ponto in grupo_custos:
            if any(np.all(outro <= ponto) for outro in candidatos):
                continue
            candidatos = [outro for outro in candidatos
                          if not (np.all(ponto <= outro) and np.any(ponto < outro))]
            candidatos.append(ponto)
        registros.append({
            "iteracao": int(iteracao),
            "hipervolume": hipervolume_2d(candidatos),
            "frente_dentro_referencia": len(candidatos),
        })
    historico = pd.DataFrame(registros)
    historico.to_csv(PASTA / "hipervolume_historico.csv", index=False)
    hv_avaliacoes = float(historico["hipervolume"].iloc[-1])
    print(f"hv recalculado pelas avaliacoes acumuladas: {hv_avaliacoes:.12e}")
    if not np.isclose(hv_avaliacoes, hv_pareto, rtol=1e-8, atol=1e-14):
        print("a frente em solucoes.csv difere da melhor frente das avaliacoes acumuladas.")

    contribs = []
    for indice, linha in pareto.iterrows():
        demais = pareto.drop(index=indice)
        contrib = hv_pareto - hipervolume_2d(filtrar_pontos_na_referencia(demais))
        contribs.append({
            "solucao": linha["solucao"] if "solucao" in linha.index else str(indice),
            "f1_score": float(linha["f1_score"]),
            "fpr": float(linha["fpr"]),
            "n_atributos": int(linha["n_atributos"]),
            "contribuicao_exclusiva_hv": contrib,
        })
    resumo = pd.DataFrame(contribs)
    resumo["hipervolume_total"] = hv_pareto
    resumo["referencia_f1"] = REF_F1
    resumo["referencia_fpr"] = REF_FPR
    resumo.to_csv(PASTA / "hipervolume_resumo.csv", index=False)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.9), layout="constrained")
    ax = eixos[0]
    ordenada = pareto.sort_values("fpr")
    x = ordenada["fpr"].to_numpy()
    y = ordenada["f1_score"].to_numpy()
    for i in range(len(x)):
        x_final = x[i + 1] if i < len(x) - 1 else REF_FPR
        ax.fill_between([x[i], x_final], REF_F1, y[i], color="#3878a8", alpha=0.22)
        ax.plot([x[i], x_final], [y[i], y[i]], color="#3878a8", lw=1.5)
    ax.scatter(x, y, color="#163d62", zorder=5, s=45, label="5 solucoes de Pareto")
    for _, linha in ordenada.iterrows():
        if "solucao" in linha.index:
            ax.annotate(str(linha["solucao"]), (linha["fpr"], linha["f1_score"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.scatter([REF_FPR], [REF_F1], marker="x", s=70, color="#bd502e", label="Referencia")
    ax.set(xlabel="FPR (minimizar)", ylabel="F1-score (maximizar)",
           title="Area dominada: hipervolume")
    ax.set_xlim(0.00155, REF_FPR * 1.035)
    ax.set_ylim(REF_F1 - .0002, .9942)
    ax.grid(alpha=.16)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = eixos[1]
    ax2.plot(historico["iteracao"], historico["hipervolume"], lw=2, color="#3878a8")
    ax2.axhline(hv_pareto, ls="--", lw=1, color="#bd502e", label="HV final")
    ax2.set(xlabel="Iteracao", ylabel="Hipervolume (area)",
            title="Evolucao do hipervolume durante a busca")
    ax2.grid(alpha=.2)
    ax2.legend()
    fig.suptitle("MOGWO | 2 objetivos: maximizar F1-score e minimizar FPR", fontsize=12)
    fig.savefig(PASTA / "hipervolume.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Arquivos gerados: hipervolume.png, hipervolume_historico.csv, hipervolume_resumo.csv")


if __name__ == "__main__":
    main()
