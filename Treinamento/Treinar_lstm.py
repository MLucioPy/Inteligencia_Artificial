import os
# 1. BLOQUEIO DE AVISOS DO TENSORFLOW (Adicionar antes de importar as bibliotecas)
# O nível '3' desativa avisos informativos, mensagens de compilação e alertas de GPU do Windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Bloqueia aviso GPU
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 2. Carregar o dataset preparado e higienizado
try:
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
except FileNotFoundError:
    print("⚠️ Erro: O arquivo 'DataSet_Preparado.csv' não foi localizado no diretório.")
    exit()

# 3. Separar as variáveis preditoras (Inputs) e a variável alvo (Target)
X_raw = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']].values
y = df['Nivel Futuro'].values

# 4. Divisão Cronológica Rígida (80% para Treino e 20% para Teste)
# Mantém a linha contínua do tempo para simular o comportamento real na planta
split_index = int(len(df) * 0.8)

X_train_seq, X_test_seq = X_raw[:split_index], X_raw[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 5. REFORMATAR PARA O PADRÃO 3D EXIGIDO PELA LSTM
# Redimensionamento estrutural: [Amostras, 1 Passo de Tempo, 4 Variáveis]
X_train = X_train_seq.reshape((X_train_seq.shape[0], 1, X_train_seq.shape[1]))
X_test = X_test_seq.reshape((X_test_seq.shape[0], 1, X_test_seq.shape[1]))

print("=========================================================")
print("          INICIANDO O TREINAMENTO DA REDE LSTM           ")
print("=========================================================")
print(f"• Amostras de Treino: {X_train.shape[0]} sequências")
print(f"• Amostras de Teste:  {X_test.shape[0]} sequências")
print("• Arquitetura: Input Layer + LSTM (50 neurónios) + Dense")
print("• Plataforma de Execução: CPU Otimizado (Ambiente Seguro)")
print("=========================================================\n")

# 6. Construção da Arquitetura da Rede Neural (Padrão Moderno Keras 3)
model = Sequential()

# Camada de Entrada explícita para eliminar avisos de obsolescência de código
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# Camada Recorrente LSTM pura com função de ativação linear retificada (ReLU)
model.add(LSTM(50, activation='relu'))

# Camada de Saída linear (1 único neurónio para estimar o valor contínuo do nível)
model.add(Dense(1))

# Compilação do modelo definindo o Erro Quadrático Médio (MSE) como função de perda
model.compile(optimizer='adam', loss='mse')

# 7. Execução do Algoritmo de Aprendizado (Ajuste de Pesos Sinápticos)
# Treinamento por 10 épocas completas processando mini-lotes de 32 em 32 linhas
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=1
)

print("\n-> Treinamento da Rede Neural de Deep Learning concluído!")
print("=========================================================\n")

# 8. Avaliação do Poder Preditivo sobre Dados Inéditos (Conjunto de Teste)
y_pred = model.predict(X_test, verbose=0).flatten()

# Cálculo das métricas oficiais de regressão para a sua Etapa 8
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=========================================================")
print("       MÉTRICAS OFICIAIS DE DESEMPENHO DA LSTM
print("=========================================================")
print(f"• MAE (Erro Médio Absoluto): {mae:.4f} %")
print(f"• RMSE (Erro Quadrático Médio): {rmse:.4f} %")
print(f"• Coeficiente R² (Ajuste Global): {r2:.4f}")
print("=========================================================\n")

# 9. Exportação da Rede Neural para o Formato Nativo Moderno
nome_modelo = 'modelo_lstm.keras'
model.save(nome_modelo)

print(f"🎉 Sucesso! O modelo de Deep Learning foi salvo como '{nome_modelo}'.")
