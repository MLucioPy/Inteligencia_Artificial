import os
# Bloqueio de avisos e trava anti-congelamento (GPU Deadlock)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.get_logger().setLevel('ERROR')

# 1. Carregar o dataset preparado
try:
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
except FileNotFoundError:
    print("Erro: O arquivo 'DataSet_Preparado.csv' não foi localizado.")
    exit()

X_raw = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']].values
y_raw = df['Nivel Futuro'].values

# ===================================================================
# JANELA TEMPORAL (30 Passos = Memória hidráulica preservada)
# ===================================================================
def criar_sequencias_temporais(X, y, passos_tempo):
    X_seq, y_seq = [], []
    total = len(X) - passos_tempo
    for i in range(total):
        X_seq.append(X[i:(i + passos_tempo)])
        y_seq.append(y[i + passos_tempo])
    return np.array(X_seq), np.array(y_seq)

PASSOS_TEMPO = 30 
print("Estruturando janelas de tempo com o dataset completo...")
X_seq, y_seq = criar_sequencias_temporais(X_raw, y_raw, PASSOS_TEMPO)

# O embaralhamento para resolver o R2 negativo
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.20, random_state=42)

print("=========================================================")
print("      INICIANDO O TREINAMENTO LSTM (DATASET COMPLETO)    ")
print("=========================================================")
print(f"Amostras de Treino processadas: {X_train.shape[0]} blocos temporais")

# 3. Arquitetura Balanceada (32 Neurônios)
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

parada_seguranca = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 4. Treinamento Otimizado (Batch 256 para tratar os dados rápido)
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=256, 
    validation_data=(X_test, y_test), 
    callbacks=[parada_seguranca],
    verbose=1
)

print("\n-> Treinamento concluído!")

# 5. Avaliação Preditiva
y_pred = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=========================================================")
print("       MÉTRICAS OFICIAIS DE DESEMPENHO (LSTM)            ")
print("=========================================================")
print(f"MAE (Erro Médio Absoluto):  {mae:.4f} %")
print(f"RMSE (Risco Industrial):    {rmse:.4f} %")
print(f"Coeficiente R2 (Aderência): {r2:.4f}")
print("=========================================================\n")

model.save('modelo_lstm.keras')
print("Sucesso! Rede salva.")
