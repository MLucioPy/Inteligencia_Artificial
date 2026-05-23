import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Carregar o dataset preparado
try:
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
except FileNotFoundError:
    print("⚠️ Erro: O arquivo 'DataSet_Preparado.csv' não foi encontrado nesta pasta.")
    exit()

# 2. Separar as variáveis preditoras (Inputs) e a variável alvo (Target)
X = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']]
y = df['Nivel Futuro']

# 3. Divisão Aleatória (80% para Treino e 20% para Teste)
# Como o RF não depende de inércia temporal, o embaralhamento (random_state) 
# garante que o modelo seja testado em todas as fases do tanque (subida, descida e estabilização).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("=========================================================")
print("        INICIANDO O TREINAMENTO RANDOM FOREST            ")
print("=========================================================")
print(f"• Amostras de Treino: {X_train.shape[0]} linhas")
print(f"• Amostras de Teste:  {X_test.shape[0]} linhas")
print("• Arquitetura: Comitê de 100 Árvores de Decisão")
print("Aguarde, processando os nós no processador (CPU)...")

# 4. Instanciar e Treinar o Modelo
# modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# Adicionamos o max_depth=15 (profundidade máxima) e min_samples_leaf=4
modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

print("-> Treinamento concluído com sucesso!")
print("=========================================================\n")

# 5. Auditoria de Desempenho com Dados Inéditos
y_pred = modelo_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=========================================================")
print("             MÉTRICAS OFICIAIS (RANDOM FOREST)           ")
print("=========================================================")
print(f"• MAE (Erro Linear):      {mae:.4f} %")
print(f"• RMSE (Risco de Planta): {rmse:.4f} %")
print(f"• R² Score (Aderência):   {r2:.4f}")
print("=========================================================\n")

# 6. Exportação do Modelo
nome_arquivo_modelo = 'modelo_random_forest.pkl'
joblib.dump(modelo_rf, nome_arquivo_modelo)

print(f"🎉 Sucesso! O modelo foi salvo como '{nome_arquivo_modelo}'.")
