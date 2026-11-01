from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PASTA = Path(__file__).resolve().parent

with open(PASTA / 'pareto.pkl', 'rb') as arquivo:
    pareto = pickle.load(arquivo)

if not pareto:
    raise ValueError('cade as solucoes')

registros = []
for i, sol in enumerate(pareto, start=1):
    mascara = ''.join(map(str, np.asarray(sol['posicao'], dtype=int)))
    custo = np.asarray(sol['custo'], dtype=float)
    registros.append({
        'rotulo': f'P{i:02d}',
        'mascara': mascara,
        'f1': 1.0 - custo[0],
        'fpr': custo[1],
    })

# Se solucoes.csv existir, os rótulos serão os mesmos usados no teste.
if (PASTA / 'solucoes.csv').exists():
    solucoes = pd.read_csv(PASTA / 'solucoes.csv', dtype={'mascara': str})
    ids = dict(zip(solucoes['mascara'].str.strip(), solucoes['solucao']))
    for registro in registros:
        registro['rotulo'] = ids.get(registro['mascara'], registro['rotulo'])

dados = pd.DataFrame(registros).sort_values('fpr').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(dados['fpr'], dados['f1'], '--', linewidth=1.2, alpha=0.65)
ax.scatter(dados['fpr'], dados['f1'], s=95, zorder=3, label='Soluções não dominadas')

for i, sol in dados.iterrows():
    ax.annotate(sol['rotulo'], (sol['fpr'], sol['f1']),
                xytext=(7, 9 if i % 2 == 0 else -16),
                textcoords='offset points', fontsize=10)

ax.set_title('Fronteira de Pareto — validação interna', fontsize=14)
ax.set_xlabel('Taxa de falsos positivos (FPR) ↓')
ax.set_ylabel('F1-score ↑')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.grid(True, alpha=0.25)
ax.legend(loc='best')
ax.margins(x=0.15, y=0.25)
fig.tight_layout()
fig.savefig(PASTA / 'grafico_pareto.png', dpi=300, bbox_inches='tight')
fig.savefig(PASTA / 'grafico_pareto.pdf', bbox_inches='tight')
print(f'Gráfico gerado com {len(dados)} soluções: grafico_pareto.png e grafico_pareto.pdf')
plt.show()
