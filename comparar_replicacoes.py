"""Compara as frentes das execuções seed42 e seed123 na mesma validação."""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAMINHOS = {
    42: 'arquivo_pareto_corrigido.pkl',
    123: 'arquivo_pareto_seed123.pkl',
}


def carregar(caminho):
    with open(caminho, 'rb') as arq:
        dados = pickle.load(arq)
    linhas = []
    for s in dados:
        custo = np.asarray(s['custo'])
        linhas.append({
            'f1': float(1 - custo[0]),
            'fpr': float(custo[1]),
            'atributos': int(np.asarray(s['posicao']).sum()),
        })
    return pd.DataFrame(linhas)


fig, ax = plt.subplots(figsize=(9, 6))
resumos = []
for seed, arquivo in CAMINHOS.items():
    if not os.path.isfile(arquivo):
        print(f'Arquivo ausente: {arquivo}. Pulei essa execução.')
        continue
    df = carregar(arquivo)
    pares = df.drop_duplicates(subset=['f1', 'fpr']).sort_values('fpr')
    ax.scatter(pares['fpr'], pares['f1'], s=35, label=f'Semente {seed} ({len(pares)} pontos)')
    melhor = df.loc[df['f1'].idxmax()]
    menor_fpr = df.loc[df['fpr'].idxmin()]
    resumos.append({
        'semente': seed,
        'subconjuntos': len(df),
        'pontos_distintos': len(pares),
        'melhor_f1': melhor['f1'],
        'fpr_do_melhor_f1': melhor['fpr'],
        'atributos_do_melhor_f1': int(melhor['atributos']),
        'menor_fpr': menor_fpr['fpr'],
        'subconjuntos_f1_90_mais': int((df['f1'] >= 0.90).sum()),
    })

if not resumos:
    raise SystemExit('Nenhum arquivo de resultados encontrado.')
ax.set_xlabel('Taxa de falsos positivos (FPR)')
ax.set_ylabel('F1-score')
ax.set_title('Soluções não dominadas: comparação entre sementes')
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('comparacao_frentes_sementes.png', dpi=200)
pd.DataFrame(resumos).to_csv('resumo_replicacoes.csv', index=False)
print(pd.DataFrame(resumos).to_string(index=False))
print('Salvos: comparacao_frentes_sementes.png e resumo_replicacoes.csv')
