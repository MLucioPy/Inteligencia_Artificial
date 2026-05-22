import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Carregar o dataset preparado e higienizado
try:
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
except FileNotFoundError:
    print("⚠️ Erro: O arquivo 'DataSet_Preparado.csv' não foi encontrado nesta pasta.")
    exit()

# 2. Separar as variáveis preditoras (Inputs) e a variável alvo (Target)
X = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']]
y = df['Nivel Futuro']

# 3. Divisão Cronológica (80% para Treino e 20% para Teste)
# Mantém a ordem do tempo para simular a IA operando de verdade no chão de fábrica
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("=========================================================")
print("        INICIANDO O TREINAMENTO RANDOM FOREST        ")
print("=========================================================")
print(f"• Amostras de Treino: {X_train.shape[0]} linhas")
print(f"• Amostras de Teste:  {X_test.shape[0]} linhas")
print("• Algoritmo: Random Forest Regressor (100 árvores)")
print("Aguarde, processando nós de decisão...")

# 4. Instanciar e Treinar o Modelo
# n_jobs=-1 usa todos os núcleos do processador do novo PC para acelerar o treino
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

print("-> Treinamento concluído com sucesso!")
print("=========================================================\n")

# 5. Avaliação de Desempenho com Dados Inéditos (Conjunto de Teste)
y_pred = modelo_rf.predict(X_test)

# Cálculo das métricas de regressão
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=========================================================")
print("             MÉTRICAS DE DESEMPENHO (ETAPA 5)            ")
print("=========================================================")
print(f"• MAE (Erro Médio Absoluto): {mae:.4f} %")
print(f"• RMSE (Erro Quadrático Médio): {rmse:.4f} %")
print(f"• Coeficiente R² (Ajuste Global): {r2:.4f}")
print("=========================================================\n")

# 6. Exportação do Modelo para o Dashboard Streamlit
nome_arquivo_modelo = 'modelo_random_forest.pkl'
joblib.dump(modelo_rf, nome_arquivo_modelo)

print(f"🎉 Sucesso! O modelo de IA foi compactado e salvo como '{nome_arquivo_modelo}'.")
