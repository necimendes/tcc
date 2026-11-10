from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PASTA = Path(__file__).resolve().parent

with (PASTA / 'pareto.pkl').open('rb') as arquivo:
    pareto = pickle.load(arquivo)

if len(pareto) != 5:
    raise ValueError(f'Esperadas cinco soluções de Pareto; encontradas {len(pareto)}.')

pontos = []
for i, sol in enumerate(pareto, start=1):
    custo = np.asarray(sol['custo'], dtype=float)
    pontos.append({
        'id': f'P{i:02d}',      
        'fpr': float(custo[1]),
        'f1': float(1.0 - custo[0]),
        'atributos': int(np.count_nonzero(sol['posicao'])),
    })

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.scatter(
    [p['fpr'] for p in pontos],
    [p['f1'] for p in pontos],
    s=100, marker='o', color='tab:blue', zorder=3,
    label='Soluções da fronteira de Pareto',
)

deslocamentos = {
    'P01': (-39, 11),
    'P02': (8, -12),
    'P03': (-37, 12),
    'P04': (8, -14),
    'P05': (8, 10),
}
for p in pontos:
    ax.annotate(
        p['id'], (p['fpr'], p['f1']),
        xytext=deslocamentos[p['id']], textcoords='offset points',
        fontsize=10, fontweight='medium',
    )

ax.set_title('Fronteira de Pareto — validação interna', fontsize=14)
ax.set_xlabel('Taxa de falsos positivos (FPR) ↓')
ax.set_ylabel('F1-score ↑')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.grid(True, alpha=0.25)
ax.legend(loc='best')
ax.margins(x=0.18, y=0.23)
fig.tight_layout()
fig.savefig(PASTA / 'grafico_pareto.png', dpi=300, bbox_inches='tight')
fig.savefig(PASTA / 'grafico_pareto.pdf', bbox_inches='tight')
plt.close(fig)

print('Soluções do gráfico (na ordem da tabela):')
for p in pontos:
    print(f"{p['id']}: F1={p['f1']:.6f}; FPR={p['fpr']:.6f}; atributos={p['atributos']}")
print('Arquivos gerados: grafico_pareto.png / .pdf')
