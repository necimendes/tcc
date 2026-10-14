import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report

# Recarrega o dataset
CAMINHO_DATASET = 'data/MachineLearningCSV/MachineLearningCVE'

dfs = []
for arquivo in sorted(os.listdir(CAMINHO_DATASET)):
    if arquivo.endswith('.csv'):
        df_temp = pd.read_csv(os.path.join(CAMINHO_DATASET, arquivo), low_memory=False)
        dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df[' Label'] = df[' Label'].apply(lambda x: 0 if x.strip() == 'BENIGN' else 1)

X = df.drop(columns=[' Label'])
y = df[' Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dados recarregados.")

# Carrega o arquivo de Pareto
with open('arquivo_pareto.pkl', 'rb') as f:
    arquivo_pareto = pickle.load(f)

# Pega o subconjunto de maior F1
f1s = [1 - p['custo'][0] for p in arquivo_pareto]
melhor = arquivo_pareto[np.argmax(f1s)]
features_selecionadas = np.where(melhor['posicao'] == 1)[0]

nomes = ['Destination Port','Flow Duration','Total Fwd Packets','Total Backward Packets',
         'Total Length Fwd Packets','Total Length Bwd Packets','Fwd Packet Length Max',
         'Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std',
         'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean',
         'Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean',
         'Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean',
         'Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean',
         'Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags',
         'Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length',
         'Fwd Packets/s','Bwd Packets/s','Min Packet Length','Max Packet Length',
         'Packet Length Mean','Packet Length Std','Packet Length Variance',
         'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count',
         'ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count',
         'Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size',
         'Avg Bwd Segment Size','Fwd Header Length','Fwd Avg Bytes/Bulk',
         'Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk',
         'Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',
         'Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes',
         'Init Win Bytes Forward','Init Win Bytes Backward','Act Data Pkt Fwd',
         'Min Seg Size Forward','Active Mean','Active Std','Active Max','Active Min',
         'Idle Mean','Idle Std','Idle Max','Idle Min']

feats_nomes = [nomes[i] for i in features_selecionadas]
print(f"\nFeatures selecionadas ({len(features_selecionadas)}): {feats_nomes}")

# Treina no conjunto de treino COMPLETO com as features selecionadas
X_tr = X_train_scaled[:, features_selecionadas]
X_te = X_test_scaled[:, features_selecionadas]

print("\nTreinando k-NN no conjunto de treino completo...")
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
knn.fit(X_tr, y_train)

print("Avaliando no conjunto de teste...")
y_pred = knn.predict(X_te)

f1_final = f1_score(y_test, y_pred, zero_division=0)
TN = np.sum((y_pred == 0) & (y_test.values == 0))
FP = np.sum((y_pred == 1) & (y_test.values == 0))
fpr_final = FP / (FP + TN) if (FP + TN) > 0 else 0.0

print(f"\n=== RESULTADO FINAL NO CONJUNTO DE TESTE ===")
print(f"F1-score:  {f1_final:.4f}")
print(f"FPR:       {fpr_final:.4f}")
print(f"Features:  {len(features_selecionadas)} de 78")
print(f"\n=== COMPARAÇÃO COM FF-WMOGWO ===")
print(f"FF-WMOGWO:     F1=0.9730 | 11 features")
print(f"Seu resultado: F1={f1_final:.4f} | FPR={fpr_final:.4f} | {len(features_selecionadas)} features")
print(f"\n=== RELATÓRIO COMPLETO ===")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Malicioso']))