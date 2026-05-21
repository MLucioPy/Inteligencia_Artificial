import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Carregar o novo dataset bruto fornecido
df = pd.read_csv('Dataset_Interpolado_Limites.csv', sep=';')

# Converter TimeStamp para o formato de data/hora real para proteção de tempo
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)

# 2. Definir o horizonte de previsão (5 minutos no futuro = 30 linhas de 10s)
minutos_futuro = 5 
passos_por_minuto = 60 / 10
linhas_deslocamento = int(minutos_futuro * passos_por_minuto)

# 3. Calcular o deslocamento temporal e a suavização de ruído
df['Tempo Futuro Esperado'] = df['TimeStamp'].shift(-linhas_deslocamento)
df['Diferenca_Segundos'] = (df['Tempo Futuro Esperado'] - df['TimeStamp']).dt.total_seconds()

# Suavização de ruído de alta frequência (Média móvel de 30s centralizada)
nivel_suavizado = df['Nivel Atual'].rolling(window=3, center=True).mean().fillna(df['Nivel Atual'])
df['Nivel Futuro'] = nivel_suavizado.shift(-linhas_deslocamento)

# Cortar conexões inválidas em paradas prolongadas de planta (Gaps > 5 min)
df.loc[df['Diferenca_Segundos'] > 310, 'Nivel Futuro'] = np.nan

# ==============================================================================
# INJEÇÃO DAS CONSIDERAÇÕES OPERACIONAIS (RESTRIÇÕES FÍSICAS DA PLANTA)
# ==============================================================================
print("Aplicando regras de restrição física ao processo...")
tol_zero = 0.01

# --- CONSIDERAÇÃO 3: MANOBRA MANUAL (Distúrbio Não Medido) ---
linhas_manobra_manual = (
    (df['Velocidade Inversor Entrada'] <= tol_zero) & 
    (df['Vazao Entrada'] <= tol_zero) & 
    (df['Vazao Saida'] <= tol_zero) & 
    (df['Nivel Futuro'] < df['Nivel Atual'] - 1.5)
)
qtd_removida_manual = linhas_manobra_manual.sum()
df = df[~linhas_manobra_manual] # Remove as linhas de manobra manual

# --- CONSIDERAÇÃO 1: BOMBA E VAZÃO DE ENTRADA ZERADAS ---
filtro_subida_indevida = (
    (df['Velocidade Inversor Entrada'] <= tol_zero) & 
    (df['Vazao Entrada'] <= tol_zero) & 
    (df['Nivel Futuro'] > df['Nivel Atual'] + 1.5)
)
df.loc[filtro_subida_indevida, 'Nivel Futuro'] = df['Nivel Atual']

# --- CONSIDERAÇÃO 2: VAZÃO DE SAÍDA ZERADA ---
filtro_queda_indevida = (
    (df['Vazao Saida'] <= tol_zero) & 
    (df['Nivel Futuro'] < df['Nivel Atual'] - 0.5)
)
df.loc[filtro_queda_indevida, 'Nivel Futuro'] = df['Nivel Atual']

print(f"-> Sucesso: {qtd_removida_manual} amostras de manobras manuais foram eliminadas.")
# ==============================================================================

# 5. Remover linhas com NaNs (trecho final do shift e gaps)
df.dropna(subset=['Nivel Futuro'], inplace=True)

# 6. Remover colunas de controle temporal e texto
df.drop(columns=['TimeStamp', 'Tempo Futuro Esperado', 'Diferenca_Segundos'], inplace=True)

# 7. Normalização das entradas (Variável Alvo 'Nivel Futuro' fica na escala original)
colunas_entrada = ['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']
scaler = MinMaxScaler()
df[colunas_entrada] = scaler.fit_transform(df[colunas_entrada])

# 8. Salvar o dataset balanceado e blindado contra anomalias
df.to_csv('DataSet_Preparado.csv', sep=';', index=False)
print(f"Total final de amostras consistentes: {df.shape[0]}")
