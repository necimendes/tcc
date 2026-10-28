"""Extrai alternativas viaveis diferentes de um CSV de avaliacoes do MOGWO.
Nao as rotula como fronteira de Pareto; solucao dominada tambem pode ser viavel.
Uso: python selecionar_solucoes_viaveis.py
"""
import numpy as np
import pandas as pd

ARQUIVO_ENTRADA = 'avaliacoes_mogwo_restrito_seed42.csv'
F1_MIN = 0.992
FPR_MAX = 0.002
QUANTIDADE_AMOSTRA = 30

avaliacoes = pd.read_csv(ARQUIVO_ENTRADA, dtype={'mascara': str})
viaveis = avaliacoes.loc[
    (avaliacoes['f1_score'] >= F1_MIN) &
    (avaliacoes['fpr'] <= FPR_MAX)
].copy()
viaveis.sort_values(['f1_score', 'fpr', 'n_atributos'],
                    ascending=[False, True, True], inplace=True, kind='stable')
viaveis = viaveis.drop_duplicates(subset=['f1_score', 'fpr'], keep='first').reset_index(drop=True)
viaveis.to_csv('catalogo_viaveis_f1_0992_seed42.csv', index=False)

# Amostra representativa pela dispersao nas tres dimensoes normalizadas.
# Isto NAO altera a regra de dominancia do MOGWO.
valores = viaveis[['f1_score', 'fpr', 'n_atributos']].to_numpy(dtype=float)
mins, maxs = valores.min(axis=0), valores.max(axis=0)
normais = (valores - mins) / np.maximum(maxs - mins, 1e-12)
# Parte do melhor F1; seleciona iterativamente o ponto mais distante dos ja selecionados.
selecionados = [0]
distancias_min = np.linalg.norm(normais - normais[0], axis=1)
for _ in range(min(QUANTIDADE_AMOSTRA, len(viaveis)) - 1):
    distancias_min[selecionados] = -1
    proximo = int(np.argmax(distancias_min))
    selecionados.append(proximo)
    distancias_min = np.minimum(distancias_min, np.linalg.norm(normais - normais[proximo], axis=1))

amostra = viaveis.iloc[selecionados].sort_values(
    ['n_atributos', 'f1_score'], ascending=[True, False]
).copy()
amostra.to_csv('amostra_30_solucoes_viaveis_seed42.csv', index=False)
print(f'Mascaras com F1 >= {F1_MIN} e FPR <= {FPR_MAX}: ',
      ((avaliacoes.f1_score >= F1_MIN) & (avaliacoes.fpr <= FPR_MAX)).sum())
print(f'Pares F1/FPR distintos e viaveis: {len(viaveis)}')
print(f'Pontos na amostra representativa: {len(amostra)}')
print('Faixa de atributos:', amostra.n_atributos.min(), 'a', amostra.n_atributos.max())

# Gráfico final: todas as combinações viáveis e as 30 soluções selecionadas.
# Não conectar os pontos: a amostra não é uma fronteira de Pareto estrita.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Pontos de referência: todas as combinações distintas que passaram nos filtros.
ax.scatter(
    viaveis['fpr'], viaveis['f1_score'],
    color='lightgray', s=27, alpha=0.8,
    label=f'Combinações viáveis distintas ({len(viaveis)})',
    zorder=1,
)

# Pontos selecionados: cores representam o número de atributos.
pontos = ax.scatter(
    amostra['fpr'], amostra['f1_score'],
    c=amostra['n_atributos'], cmap='viridis',
    s=85, alpha=0.95, edgecolors='black', linewidths=0.5,
    label=f'Soluções selecionadas ({len(amostra)})',
    zorder=2,
)

barra = fig.colorbar(pontos, ax=ax, pad=0.02)
barra.set_label('Número de atributos selecionados')

ax.set_xlabel('Taxa de falsos positivos (FPR) — menor é melhor')
ax.set_ylabel('F1-score — maior é melhor')
ax.set_title('MOGWO: alternativas viáveis de alto desempenho (semente 42)')
ax.grid(alpha=0.25, linestyle='--')
ax.legend(loc='best')
fig.tight_layout()

CAMINHO_GRAFICO = 'grafico_solucoes_viaveis_seed42.png'
fig.savefig(CAMINHO_GRAFICO, dpi=300, bbox_inches='tight')
print(f'Gráfico salvo em: {CAMINHO_GRAFICO}')
plt.show()
