from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PASTA = Path(__file__).resolve().parent
dados = pd.read_csv(PASTA / 'teste.csv')
if len(dados)!=10 or dados['solucao'].nunique()!=10:
    raise ValueError('Este gráfico deve conter exatamente 10 soluções distintas.')
pareto = dados.loc[dados['tipo'].astype(str).str.strip().str.casefold().eq('pareto')]
adicionais = dados.loc[~dados.index.isin(pareto.index)]
if len(pareto)!=5 or len(adicionais)!=5:
    raise ValueError('São esperadas cinco soluções Pareto e cinco alternativas.')

fig, ax = plt.subplots(figsize=(9,5.8))
ax.scatter(adicionais['fpr_teste'],adicionais['f1_teste'],s=75,marker='o',
           color='tab:orange',alpha=.88,zorder=2,
           label='Alternativas compactas (5)')
ax.scatter(pareto['fpr_teste'],pareto['f1_teste'],s=100,marker='o',
           color='tab:blue',edgecolors='black',linewidths=.7,zorder=3,
           label='Soluções da fronteira de Pareto (5)')
deslocamentos={
    'P01':(-42,14),'P02':(8,-17),'P03':(8,-17),
    'P04':(-5,17),'P05':(-48,-13)
}
for _,sol in pareto.iterrows():
    ax.annotate(sol['solucao'],(sol['fpr_teste'],sol['f1_teste']),
                xytext=deslocamentos.get(sol['solucao'],(8,10)),
                textcoords='offset points',fontsize=10)
ax.set_title('Desempenho das soluções no teste externo',fontsize=14)
ax.set_xlabel('Taxa de falsos positivos (FPR) ↓')
ax.set_ylabel('F1-score ↑')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
ax.grid(True,alpha=.25)
ax.legend(loc='best')
ax.margins(x=.20,y=.22)
fig.tight_layout()
fig.savefig(PASTA/'grafico_teste.png',dpi=300,bbox_inches='tight')
fig.savefig(PASTA/'grafico_teste.pdf',bbox_inches='tight')
plt.close(fig)
