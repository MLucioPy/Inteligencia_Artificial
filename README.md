# Inteligencia_Artificial
Predição Futura do Nível de Tanques Industriais utilizando Inteligência Artificial.

# Random Forest vs. Rede Neural LSTM

Este repositório contém o ecossistema completo de desenvolvimento de um **Gêmeo Digital de Processos** para a predição e o monitoramento preditivo do nível de um reservatório industrial. O projeto foi estruturado com foco em rigor científico e aplicabilidade prática em sistemas de automação de chão de fábrica, servindo como base para o Trabalho Acadêmico de Inteligência Artificial do **Instituto Federal de São Paulo (IFSP)**.

O objetivo do sistema é prever o comportamento dinâmico do volume interno do tanque em um horizonte preditivo de **5 minutos no futuro (T + 5 min)**. Isso permite tomadas de decisão antecipadas por parte das equipes de instrumentação e controle, mitigando riscos de transbordamento ou desabastecimento.

---

## 🎯 Escopo Técnico do Projeto

O ecossistema realiza o confronto direto entre duas abordagens tecnológicas consagradas no cenário de Aprendizado de Máquina e Aprendizado Profundo:

1. **Random Forest Regressor:** Abordagem baseada em comitês de árvores de decisão. Apresenta alta robustez contra multicolinearidade e grande estabilidade em transições rápidas de sinais.
2. **Rede Neural Recorrente LSTM (Long Short-Term Memory):** Abordagem baseada em *Deep Learning*. É projetada estruturalmente com blocos de memória capazes de reter dependências e a inércia temporal de longo prazo inerente ao transporte hidráulico de fluidos.

### Variáveis Monitoradas (Inputs do Modelo)
* **Nível Atual do Tanque (%)**
* **Vazão de Entrada (L/h)**
* **Vazão de Saída / Consumo Downstream (L/h)**
* **Velocidade do Inversor de Frequência da Bomba (%)**

### Variável Alvo (Target)
* **Nível Futuro do Tanque (%)** em T + 30 passos (equivalente a 5 minutos à frente, com taxa de amostragem de 10 segundos).

---

## 📂 Arquitetura do Repositório

O projeto está subdividido em scripts modulares que organizam o ciclo de vida dos dados de forma limpa e sequencial:

* 📄 `preparar_dataset.py`: Realiza a engenharia de recursos, suavização de ruídos de alta frequência através de média móvel e injeção de restrições físicas industriais (detecção e remoção de manobras manuais de dreno não medidas). Aplica normalização Min-Max nas entradas.
* 📄 `analise_dataset_preparado.py`: Executa a auditoria de dados pós-processamento. Avalia a assimetria (*skewness*), a matriz de correlação de Pearson e quantifica o ganho de informação não-linear (*feature importance*) de cada sensor através de uma árvore base.
* 📄 `treinar_random_forest.py`: Pipeline focado na construção do modelo de árvores, aplicando divisão cronológica estrita (80% treino / 20% teste) para respeitar a linha contínua do tempo. Exporta o arquivo compactado `modelo_random_forest.pkl`.
* 📄 `treinar_lstm.py`: Pipeline focado na estruturação do modelo de Deep Learning no formato 3D exigido pelo TensorFlow. Implementa a sintaxe moderna do Keras 3 (camada `Input` explícita) e salva a rede como `modelo_lstm.keras`.
* 📄 `app.py`: Interface gráfica responsiva e de alta performance criada em **Streamlit**. Permite a simulação em tempo real através de sliders, apresenta tomada de decisão adaptativa e exibe uma área nobre focada em auditoria estatística visual.

---

## ⚙️ Indicadores de Desempenho Técnico (Métricas de Validação)

Para garantir a confiabilidade operacional exigida pelo setor industrial e pela banca examinadora, os modelos são submetidos a três auditorias estatísticas complementares:

1. **MAE (Erro Médio Absoluto):** Analisa a magnitude média linear dos erros residuais, apontando o desvio padrão da rotina de controle permanente da planta.
2. **RMSE (Erro Quadrático Médio):** Indicador crítico de processo. Como eleva os desvios ao quadrado antes de extrair a média, ele penaliza severamente grandes erros transientes. Valores baixos de RMSE garantem que o modelo é seguro e está livre de falhas preditivas graves que colocariam a integridade mecânica do tanque em risco.
3. **R² Score (Coeficiente de Determinação):** Quantifica o nível de aderência e ajuste geométrico global do modelo preditivo em relação aos dados reais de chão de fábrica.

---

## 🚀 Como Executar o Projeto Localmente

### 1. Instalação das Dependências
Certifique-se de ter o Python 3.12 (ou superior) instalado. Abra o terminal na pasta raiz do projeto e execute o comando abaixo para instalar o ecossistema necessário:

```bash
pip install tensorflow-cpu streamlit scikit-learn pandas numpy matplotlib joblib

### Passo 01: Tratamento e Higienização dos Dados Brutos:

```bash
python preparar_dataset.py

### Passo 02: Auditoria Estatística e Relevância de Recursos:

```bash
python analise_dataset_preparado.py

### Passo 03: Treinamento e Exportação do Modelo Random Forest:

```bash
python treinar_random_forest.py

### Passo 04: Treinamento e Ajuste de Pesos Sinápticos da LSTM:

```bash
python treinar_lstm.py

### Passo 05: Inicializando o Dashboard:

```bash
streamlit run app.py

## 🎓 Finalidade Acadêmica e Créditos

Este projeto foi desenvolvido como artefato prático-tecnológico para a defesa pública do Trabalho de Conclusão da Matéria de Inteligência Artificial do Instituto Federal de São Paulo (IFSP).

A aplicação simula com sucesso uma arquitetura industrial em ambiente computacional otimizado para CPU, ideal para demonstrações em tempo real de integração entre Engenharia de Dados e Inteligência Artificial Aplicada ao Chão de Fábrica.
