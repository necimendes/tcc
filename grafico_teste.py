from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PASTA = Path(__file__).resolve().parent
dados = pd.read_csv(PASTA / 'teste.csv')

colunas = {'solucao', 'tipo', 'f1_teste', 'fpr_teste'}
faltantes = colunas - set(dados.columns)
if faltantes:
    raise ValueError(f'Colunas ausentes em teste.csv: {sorted(faltantes)}')

eh_pareto = dados['tipo'].astype(str).str.strip().str.casefold() == 'pareto'
adicionais = dados.loc[~eh_pareto]
pareto = dados.loc[eh_pareto]

fig, ax = plt.subplots(figsize=(9, 5.8))
ax.scatter(adicionais['fpr_teste'], adicionais['f1_teste'],
           s=60, alpha=0.75, marker='o',
           label=f'Alternativas de alto desempenho ({len(adicionais)})')
ax.scatter(pareto['fpr_teste'], pareto['f1_teste'],
           s=115, marker='D', zorder=3,
           label=f'Soluções originárias do Pareto ({len(pareto)})')

deslocamentos = {
    'S01': (-39, 20), 'S03': (9, -19), 'S06': (10, 24),
    'S09': (10, -20), 'S11': (10, 12),
}
for i, (_, sol) in enumerate(pareto.iterrows()):
    dx, dy = deslocamentos.get(str(sol['solucao']), (8, 14 if i % 2 == 0 else -17))
    ax.annotate(str(sol['solucao']),
                (sol['fpr_teste'], sol['f1_teste']),
                xytext=(dx, dy), textcoords='offset points', fontsize=10)

ax.set_title('Desempenho das soluções no teste externo', fontsize=14)
ax.set_xlabel('Taxa de falsos positivos (FPR) ↓')
ax.set_ylabel('F1-score ↑')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.grid(True, alpha=0.25)
ax.legend(loc='best')
ax.margins(x=0.13, y=0.20)
fig.tight_layout()
fig.savefig(PASTA / 'grafico_teste.png', dpi=300, bbox_inches='tight')
fig.savefig(PASTA / 'grafico_teste.pdf', bbox_inches='tight')
print(f'Gráfico gerado com {len(dados)} soluções: grafico_teste.png e grafico_teste.pdf')
plt.show()
