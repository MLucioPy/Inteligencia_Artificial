import pandas as pd

# 1. CARREGAMENTO E INTEGRIDADE TEMPORAL
# Carrega o arquivo utilizando o separador correto ponto e vírgula (;)
df = pd.read_csv('Dataset_Interpolado_Limites.csv', sep=';')

# Converte a coluna de tempo para o formato datetime nativo do Python
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)

# Calcula o intervalo de tempo entre as amostras sequenciais para validar a continuidade
diferencas_tempo = df['TimeStamp'].diff().dropna().dt.total_seconds()
gaps_detectados = (diferencas_tempo > 10).sum()

print("=========================================================")
print("             RELATÓRIO DE INTEGRIDADE DOS DADOS          ")
print("=========================================================")
print(f"Início da amostragem: {df['TimeStamp'].min()}")
print(f"Fim da amostragem:    {df['TimeStamp'].max()}")
print(f"Total de registros:   {len(df)} linhas")
print(f"Lacunas de tempo (>10s) encontradas: {gaps_detectados}\n")


# 2. ESTATÍSTICA DESCRITIVA DAS VARIÁVEIS BRUTAS
print("=========================================================")
print("            ESTATÍSTICAS DE ENGENHARIA DA PLANTA         ")
print("=========================================================")
# Gera o sumário com média, desvio padrão, valores mínimos e máximos
sumario = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']].describe()
print(sumario.to_string())
print("\n")


# 3. LÓGICA DOS ESTADOS OPERACIONAIS ESTÁTICOS
# Tolerância de calibração para o "zero" real dos sensores industriais
tol_zero = 0.1

# Condição 1: Bomba de Entrada Desligada (Inversor zerado E sem fluxo medido)
bomba_off = (df['Velocidade Inversor Entrada'] <= tol_zero) & (df['Vazao Entrada'] <= tol_zero)

# Condição 2: Consumo Bloqueado (Nenhuma vazão de saída registada a jusante)
saida_off = (df['Vazao Saida'] <= tol_zero)

# Condição 3: Planta Totalmente Parada (Falta de circulação forçada em todo o loop)
planta_off = bomba_off & saida_off

# Contagem absoluta e percentual
total_linhas = len(df)
print("=========================================================")
print("            DIAGNÓSTICO DOS ESTADOS OPERACIONAIS         ")
print("=========================================================")
print(f"Bomba de Entrada Desligada:  {bomba_off.sum()} amostras ({bomba_off.mean()*100:.2f}%)")
print(f"Vazão de Saída Zerada:       {saida_off.sum()} amostras ({saida_off.mean()*100:.2f}%)")
print(f"Planta Totalmente Parada:    {planta_off.sum()} amostras ({planta_off.mean()*100:.2f}%)\n")


# 4. MATRIZ DE CORRELAÇÃO LINEAR PURA
print("=========================================================")
print("               MATRIZ DE CORRELAÇÃO FÍSICA               ")
print("=========================================================")
# Mede o acoplamento linear entre os instrumentos de medição atuais
matriz_corr = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']].corr()
print(matriz_corr.to_string())
