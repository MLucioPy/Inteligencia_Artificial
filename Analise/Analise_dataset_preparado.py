import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# 1. Carregar o dataset preparado (dados normalizados + target real)
try:
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
except FileNotFoundError:
    print("⚠️ Erro: O arquivo 'DataSet_Preparado.csv' não foi localizado. Execute o script de preparação primeiro.")
    exit()

print("==================================================================")
print("       AUDITORIA TÉCNICO-CIENTÍFICA DO DATASET PREPARADO          ")
print("==================================================================")
print(f"• Total de Registros para Treinamento/Teste: {df.shape[0]} linhas")
print(f"• Quantidade de Variáveis Preditoras (Inputs): {df.shape[1] - 1}")
print(f"• Horizonte Temporal Mapeado: T + 5 minutos (300 segundos)")
print("==================================================================\n")

# --- 1. VERIFICAÇÃO DE ASSIMETRIA DA DISTRIBUIÇÃO (SKEWNESS) ---
print("--- 1. Perfil de Distribuição Dinâmica do Alvo (Nível Futuro) ---")
assimetria = df['Nivel Futuro'].skew()
print(f"  * Coeficiente de Assimetria de Pearson: {assimetria:.4f}")
if abs(assimetria) < 0.5:
    print("    [DIAGNÓSTICO] Distribuição Simétrica. O tanque opera de forma equilibrada")
    print("                  em toda a sua faixa útil de capacidade.")
elif assimetria > 0.5:
    print("    [DIAGNÓSTICO] Assimetria Positiva (À Direita). A base de dados concentra")
    print("                  mais registros do tanque operando em níveis baixos/médios.")
else:
    print("    [DIAGNÓSTICO] Assimetria Negativa (À Esquerda). A base de dados concentra")
    print("                  mais registros do tanque operando em níveis elevados.")
print("-" * 66)

# --- 2. DIAGNÓSTICO DE MULTICOLINEARIDADE ---
print("\n--- 2. Diagnóstico de Multicolinearidade entre Sinais ---")
corr_inv_vazao = df['Velocidade Inversor Entrada'].corr(df['Vazao Entrada'])
print(f"  * Correlação Linear (Inversor vs Vazão de Entrada): {corr_inv_vazao:.4f}")
if corr_inv_vazao > 0.85:
    print("    [ALERTA] Acoplamento crítico detectado! Há redundância de informação.")
    print("             • Impacto na LSTM: Pode causar instabilidade nos pesos iniciais.")
    print("             • Impacto no Random Forest: É imune a isso devido à seleção aleatória.")
else:
    print("    [INFO] Nível de independência de sensores aceitável para o modelo.")
print("-" * 66)

# --- 3. ANÁLISE DE CORRELAÇÃO ESPECÍFICA COM O ALVO ---
print("\n--- 3. Força de Associação Linear com o Horizonte de Previsão ---")
correlacoes = df.corr()['Nivel Futuro'].sort_values(ascending=False)
for var, val in correlacoes.items():
    if var != 'Nivel Futuro':
        # CORRIGIDO: O formato correto de preenchimento com traço é {var:-<30}
        print(f"  * {var:-<30} Força de Acoplamento: {val:+.4f}")
print("-" * 66)

# --- 4. HEURÍSTICA DE IMPORTÂNCIA NÃO-LINEAR (FEATURE IMPORTANCE) ---
print("\n--- 4. Importância Não-Linear de Recursos (Heurística de Árvores) ---")
# Isola as variáveis para rodar uma árvore de decisão de profundidade controlada
X = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']]
y = df['Nivel Futuro']

# Instancia um regressor base para calcular entropia/ganho de informação
modelo_diagnostico = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo_diagnostico.fit(X, y)

# Extrai e exibe a importância percentual de cada variável
importancias = modelo_diagnostico.feature_importances_
for col, imp in zip(X.columns, importancias):
    # CORRIGIDO: O formato correto de preenchimento com traço é {col:-<30}
    print(f"  * {col:-<30} Peso no Aprendizado: {imp*100:.2f}%")
print("==================================================================")
