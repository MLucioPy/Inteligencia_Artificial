import os
# 1. SUPRESSÃO DE LOGS E AVISOS DO SISTEMA (Deve ser a primeira instrução)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importação protegida do TensorFlow e silenciamento interno do logger do Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    tf.get_logger().setLevel('ERROR')
    TF_DISPONIVEL = True
except ModuleNotFoundError:
    TF_DISPONIVEL = False

# 2. CONFIGURAÇÃO DA PÁGINA WEB
st.set_page_config(
    page_title="- RF vs LSTM -",
    page_icon="💧",
    layout="wide"
)

# Estilização profissional em CSS para os Cards Industriais e Destaques Estatísticos
st.markdown("""
    <style>
    .main-title { font-family: 'DM Sans', sans-serif; color: #005088; text-align: center; font-size: 38px; font-weight: bold; margin-bottom: 25px; }
    .card-rf { background-color: #f0fdf4; border-left: 5px solid #16a34a; padding: 20px; border-radius: 8px; height: 100%; }
    .card-lstm { background-color: #eff6ff; border-left: 5px solid #2563eb; padding: 20px; border-radius: 8px; height: 100%; }
    .metric-val { font-size: 35px; font-weight: bold; color: #1e293b; }
    
    /* Cores Customizadas para os Containers de Métricas Individuais */
    .mae-container { background-color: #fffbeb; border: 2px dashed #fbbf24; padding: 15px; border-radius: 8px; text-align: center; }
    .mae-title { color: #d97706; font-weight: bold; font-size: 13px; letter-spacing: 1px; }
    .mae-number { font-size: 32px; font-weight: bold; color: #92400e; }
    
    .rmse-container { background-color: #fff5f5; border: 2px dashed #f87171; padding: 15px; border-radius: 8px; text-align: center; }
    .rmse-title { color: #dc2626; font-weight: bold; font-size: 13px; letter-spacing: 1px; }
    .rmse-number { font-size: 32px; font-weight: bold; color: #991b1b; }
    
    .r2-container { background-color: #f3e8ff; border: 2px dashed #c084fc; padding: 15px; border-radius: 8px; text-align: center; }
    .r2-title { color: #9333ea; font-weight: bold; font-size: 13px; letter-spacing: 1px; }
    .r2-number { font-size: 32px; font-weight: bold; color: #6b21a8; }
    
    .box-comparativo { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Painel de Monitoramento e Predição de Nível de Tanque</div>', unsafe_allow_html=True)
st.write("---")

if not TF_DISPONIVEL:
    st.error("⚠️ Erro: O TensorFlow não foi detectado no Python deste computador. Instale digitando 'pip install tensorflow-cpu' no terminal.")
    st.stop()

# 3. CONSTRUÇÃO E TREINAMENTO EM CACHE (BLINDADO CONTRA WARNINGS)
@st.cache_resource
def treinar_modelos_industriais():
    df = pd.read_csv('DataSet_Preparado.csv', sep=';')
    
    X_dados = df[['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada']].values
    y_dados = df['Nivel Futuro'].values
    
    # Criação das Janelas Temporais para a LSTM (Lookback de 6 passos = 1 minuto de histórico)
    passos_passado = 6
    X_janelas, y_alvos = [], []
    for i in range(len(X_dados) - passos_passado):
        X_janelas.append(X_dados[i:(i + passos_passado), :])
        y_alvos.append(y_dados[i + passos_passado])
        
    X_3D = np.array(X_janelas)
    y_3D = np.array(y_alvos)
    
    # Divisão Cronológica Estrita (80% Treino / 20% Teste)
    X_train_3D, X_test_3D, y_train, y_test = train_test_split(X_3D, y_3D, test_size=0.2, shuffle=False)
    
    X_train_2D = X_train_3D[:, -1, :]
    X_test_2D = X_test_3D[:, -1, :]
    
    # --- MODELO 1: Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_2D, y_train)
    y_pred_rf = rf.predict(X_test_2D)
    
    # --- MODELO 2: Rede Neural LSTM ---
    lstm = Sequential([
        Input(shape=(X_train_3D.shape[1], X_train_3D.shape[2])),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    lstm.compile(optimizer='adam', loss='mean_squared_error')
    lstm.fit(X_train_3D, y_train, epochs=15, batch_size=32, verbose=0)
    y_pred_lstm = lstm.predict(X_test_3D, verbose=0).flatten()
    
    resultados = {
        'rf': {
            'modelo': rf, 'y_pred': y_pred_rf,
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'r2': r2_score(y_test, y_pred_rf)
        },
        'lstm': {
            'modelo': lstm, 'y_pred': y_pred_lstm,
            'mae': mean_absolute_error(y_test, y_pred_lstm),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lstm)),
            'r2': r2_score(y_test, y_pred_lstm)
        }
    }
    return resultados, df, y_test

with st.spinner("⚙️ Sincronizando e calculando pesos das IAs... Por favor, aguarde."):
    resultados, df, y_test = treinar_modelos_industriais()

# 4. PAINEL LATERAL INTERATIVO
st.sidebar.header("🕹️ Painel de Controle Operacional")

opcao_modelo = st.sidebar.radio(
    "Selecione o Modo de Visualização:",
    ["Opção I: Apenas Random Forest", "Opção II: Apenas Rede LSTM", "Opção III: Comparar Ambos"]
)

st.sidebar.write("---")
st.sidebar.markdown("**Simulador de Variáveis de Processo (Sinais de 0.0 a 1.0):**")

input_nivel = st.sidebar.slider("Nível Atual do Tanque", 0.0, 1.0, float(df['Nivel Atual'].mean()), 0.01)
input_vazao_in = st.sidebar.slider("Vazão de Entrada", 0.0, 1.0, float(df['Vazao Entrada'].mean()), 0.01)
input_vazao_out = st.sidebar.slider("Vazão de Saída", 0.0, 1.0, float(df['Vazao Saida'].mean()), 0.01)
input_inversor = st.sidebar.slider("Frequência do Inversor", 0.0, 1.0, float(df['Velocidade Inversor Entrada'].mean()), 0.01)

# 5. CÁLCULO DAS PREDIÇÕES EM TEMPO REAL
dados_base_simulacao = [input_nivel, input_vazao_in, input_vazao_out, input_inversor]
predicao_rf = None
predicao_lstm = None

if "Random Forest" in opcao_modelo or "Comparar Ambos" in opcao_modelo:
    X_sim_2d = pd.DataFrame([dados_base_simulacao], columns=['Nivel Atual', 'Vazao Entrada', 'Vazao Saida', 'Velocidade Inversor Entrada'])
    predicao_rf = resultados['rf']['modelo'].predict(X_sim_2d)[0]

if "Rede LSTM" in opcao_modelo or "Comparar Ambos" in opcao_modelo:
    array_2d = np.array([dados_base_simulacao])
    X_sim_3d = np.repeat(array_2d[:, np.newaxis, :], 6, axis=1)
    predicao_lstm = resultados['lstm']['modelo'].predict(X_sim_3d, verbose=0).flatten()[0]

# 6. ARRANJO RESPONSIVO DOS CARDS DE PREDIÇÃO EM TEMPO REAL
if opcao_modelo == "Opção III: Comparar Ambos":
    col1, col2, col3 = st.columns(3)
else:
    col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Sensores Atuais")
    st.metric(label="Volume Interno Atual", value=f"{input_nivel*100:.1f} %")
    st.write(f"• Vazão Entrada: {input_vazao_in:.2f} u.n.")
    st.write(f"• Vazão Saída: {input_vazao_out:.2f} u.n.")
    st.write(f"• Inversor: {input_inversor:.2f} u.n.")

if "Random Forest" in opcao_modelo:
    with col2:
        st.markdown(f"""
            <div class="card-rf">
                <h3>🌲 Predição Atual: Random Forest Regressor</h3>
                <p>Nível estimado para daqui a 5 minutos (T + 5 min):</p>
                <div class="metric-val">{predicao_rf:.2f} %</div>
                <small style='color: #16a34a;'>Abordagem baseada em Árvores de Decisão.</small>
            </div>
        """, unsafe_allow_html=True)

elif "Rede LSTM" in opcao_modelo:
    with col2:
        st.markdown(f"""
            <div class="card-lstm">
                <h3>🧠 Predição Atual: Rede Neural LSTM</h3>
                <p>Nível estimado para daqui a 5 minutos (T + 5 min):</p>
                <div class="metric-val">{predicao_lstm:.2f} %</div>
                <small style='color: #2563eb;'>Abordagem baseada em Aprendizagem Profunda Recorrente.</small>
            </div>
        """, unsafe_allow_html=True)

elif "Comparar Ambos" in opcao_modelo:
    with col2:
        st.markdown(f"""
            <div class="card-rf">
                <h3>🌲 Predição: Random Forest</h3>
                <p>Nível estimado em T + 5 min:</p>
                <div class="metric-val">{predicao_rf:.2f} %</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="card-lstm">
                <h3>🧠 Predição: Rede LSTM</h3>
                <p>Nível estimado em T + 5 min:</p>
                <div class="metric-val">{predicao_lstm:.2f} %</div>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 7. EXIBIÇÃO DE ALTA PERFORMANCE COM MAXIMA ÊNFASE EM MAE, RMSE E R²
# ==============================================================================
st.write("##")
st.markdown("### 🏆 Rigor Estatístico: Indicadores de Validação")
st.markdown("""
A validação técnica é auditada sob três perspectivas matemáticas complementares:
* **MAE (Erro Médio Absoluto):** Mede o desvio médio linear das previsões. Como não eleva os resíduos ao quadrado, reflete o erro operacional típico esperado na rotina estável da planta.
* **RMSE (Erro Quadrático Médio):** É a métrica mestre para a segurança de processos. Por quadratizar os erros antes da média, penaliza drasticamente transientes abruptos ou falhas graves de previsão.
* **$R^2$ (Coeficiente de Determinação):** Avalia a aderência geométrica global da curva simulada em relação ao comportamento real de chão de fábrica.
""")

if "Random Forest" in opcao_modelo:
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
            <div class="mae-container">
                <div class="mae-title">📐 ERRO MÉDIO LINEAR: MAE (Random Forest)</div>
                <div class="mae-number">{resultados['rf']['mae']:.4f} %</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="rmse-container">
                <div class="rmse-title">🚨 SEGURANÇA DE PROCESSO: RMSE (Random Forest)</div>
                <div class="rmse-number">{resultados['rf']['rmse']:.4f} %</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
            <div class="r2-container">
                <div class="r2-title">📈 AJUSTE GEOMÉTRICO: R² SCORE (Random Forest)</div>
                <div class="r2-number">{resultados['rf']['r2']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

elif "Rede LSTM" in opcao_modelo:
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
            <div class="mae-container">
                <div class="mae-title">📐 ERRO MÉDIO LINEAR: MAE (Rede LSTM)</div>
                <div class="mae-number">{resultados['lstm']['mae']:.4f} %</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="rmse-container">
                <div class="rmse-title">🚨 SEGURANÇA DE PROCESSO: RMSE (Rede LSTM)</div>
                <div class="rmse-number">{resultados['lstm']['rmse']:.4f} %</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
            <div class="r2-container">
                <div class="r2-title">📈 AJUSTE GEOMÉTRICO: R² SCORE (Rede LSTM)</div>
                <div class="r2-number">{resultados['lstm']['r2']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

elif "Comparar Ambos" in opcao_modelo:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
            <div class="box-comparativo">
                <h4 style="color: #16a34a; margin-top:0; margin-bottom:15px;">🌲 Auditoria Técnica: Random Forest</h4>
                <p style="background-color: #fffbeb; padding: 10px; border-radius: 6px; border-left: 4px solid #fbbf24; color: #92400e; font-size:14px; margin-bottom:10px;">
                    • <b>Erro Médio Linear (MAE): {resultados['rf']['mae']:.4f} %</b>
                </p>
                <p style="background-color: #fee2e2; padding: 10px; border-radius: 6px; border-left: 4px solid #ef4444; color: #991b1b; font-size:14px; margin-bottom:10px;">
                    • <b>Métrica de Risco (RMSE): {resultados['rf']['rmse']:.4f} %</b>
                </p>
                <p style="background-color: #f3e8ff; padding: 10px; border-radius: 6px; border-left: 4px solid #c084fc; color: #6b21a8; font-size:14px;">
                    • <b>Ajuste de Curva (R² Score): {resultados['rf']['r2']:.4f}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="box-comparativo">
                <h4 style="color: #2563eb; margin-top:0; margin-bottom:15px;">🧠 Auditoria Técnica: Rede LSTM</h4>
                <p style="background-color: #fffbeb; padding: 10px; border-radius: 6px; border-left: 4px solid #fbbf24; color: #92400e; font-size:14px; margin-bottom:10px;">
                    • <b>Erro Médio Linear (MAE): {resultados['lstm']['mae']:.4f} %</b>
                </p>
                <p style="background-color: #fee2e2; padding: 10px; border-radius: 6px; border-left: 4px solid #ef4444; color: #991b1b; font-size:14px; margin-bottom:10px;">
                    • <b>Métrica de Risco (RMSE): {resultados['lstm']['rmse']:.4f} %</b>
                </p>
                <p style="background-color: #f3e8ff; padding: 10px; border-radius: 6px; border-left: 4px solid #c084fc; color: #6b21a8; font-size:14px;">
                    • <b>Ajuste de Curva (R² Score): {resultados['lstm']['r2']:.4f}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

st.write("##")

# 8. CONSTRUÇÃO E AJUSTE ADAPTATIVO DO GRÁFICO VALIDAÇÃO DE CHÃO DE FÁBRICA
st.subheader("📈 Análise Temporal: Confronto Dinâmico no Histórico de Teste")
quantidade_pontos = st.slider("Janela de amostragem visível no gráfico (Linha do Tempo):", 50, 500, 150)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(y_test[:quantidade_pontos], label='Nível Real (Chão de Fábrica)', color='#1e3a8a', linewidth=2.5)

if "Random Forest" in opcao_modelo or "Comparar Ambos" in opcao_modelo:
    ax.plot(resultados['rf']['y_pred'][:quantidade_pontos], 
            label='Estimativa Random Forest', color='#16a34a', linestyle='--', linewidth=2)

if "Rede LSTM" in opcao_modelo or "Comparar Ambos" in opcao_modelo:
    ax.plot(resultados['lstm']['y_pred'][:quantidade_pontos], 
            label='Estimativa Rede Neural LSTM', color='#2563eb', linestyle=':', linewidth=2.5)

ax.set_title('Comportamento dos Modelos contra os Dados Reais de Validação', fontsize=12, fontweight='bold', color='#0f172a')
ax.set_xlabel('Linha do Tempo Sequenciais (Intervalos de 10 segundos)', fontsize=10)
ax.set_ylabel('Nível (%)', fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.5)

st.pyplot(fig)

st.write("---")
st.caption(" Pós-Graduação em Inteligência Artificial - IFSP. Ambiente 100% Otimizado.")
