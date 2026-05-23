# Inteligência Artificial

# Random Forest vs. Rede Neural LSTM

👉 **[Acesso na nuvem clicando aqui!](https://tanque-ia-ifsp.streamlit.app/)** ☁️

Este repositório contém o ecossistema completo de desenvolvimento para a predição e o monitoramento do nível de um reservatório industrial. O projeto foi estruturado com foco em rigor científico e aplicabilidade prática em sistemas de automação de chão de fábrica, servindo como base para o Trabalho de Conclusão da Matéria Inteligência Artificial do **Instituto Federal de São Paulo (IFSP)**.

O objetivo do sistema é prever o comportamento dinâmico do volume interno do tanque em um horizonte preditivo de **5 minutos no futuro (T + 5 min)**. Isso permite tomadas de decisão antecipadas por parte das equipes de instrumentação e controle, mitigando riscos de transbordamento ou desabastecimento.

---

## 🎯 Escopo Técnico do Projeto

O ecossistema realiza o confronto direto entre duas abordagens tecnológicas consagradas no cenário de Aprendizado de Máquina e Aprendizado Profundo:

1. **Random Forest Regressor:** Abordagem baseada em comitês de árvores de decisão. Apresenta alta robustez contra multicolinearidade e grande estabilidade em transições rápidas de sinais.
2. **Rede Neural Recorrente LSTM (Long Short-Term Memory):** Abordagem baseada em *Deep Learning*. É projetada estruturalmente com blocos de memória capazes de reter dependências e a inércia temporal de longo prazo inerente ao transporte hidráulico de fluidos.

---

## 📊 Relatório de Integridade e Caracterização Estatística dos Dados

Os dados operacionais coletados no chão de fábrica foram auditados e caracterizados para garantir a máxima robustez e confiabilidade durante o processo de modelagem preditiva das duas inteligências artificiais.

### 1. Auditoria de Integridade e Temporalidade
* **Janela Temporal de Amostragem:** 16 de Maio de 2026 às 12:00:00 até 21 de Maio de 2026 às 11:59:50.
* **Volume do Dataset Bruto:** 43.200 registros sequenciais.
* **Frequência de Varredura:** 1 registro a cada 10 segundos (Rigidamente Estável).
* **Consistência Perfeita:** Foram identificadas e validadas 43.199 conexões temporais consecutivas exatas de 10 segundos, confirmando a ausência completa de lacunas, falhas de comunicação (*gaps*) ou perda de pacotes de dados (*data drops*).

### 2. Análise Descritiva das Variáveis de Processo
A tabela abaixo consolida o comportamento estatístico das variáveis operacionais distribuídas no dataset:

| Métrica Estatística | Nível Atual (%) | Vazão Entrada (u.n.) | Vazão Saída (u.n.) | Velocidade Inversor Entrada (u.n.) |
| :--- | :---: | :---: | :---: | :---: |
| **Média** | 63,5513% | 0,4460 | 0,4429 | 0,3811 |
| **Desvio Padrão (Std)** | 14,3577% | 0,3884 | 0,2294 | 0,3321 |
| **Valor Mínimo (Min)** | 16,9132% | 0,0000 | 0,0000 | 0,0000 |
| **Quartil 25%** | 56,1100% | 0,0000 | 0,2835 | 0,0000 |
| **Mediana (50%)** | 66,7426% | 0,5109 | 0,4461 | 0,4360 |
| **Quartil 75%** | 73,7800% | 0,8127 | 0,6179 | 0,6943 |
| **Valor Máximo (Max)** | 88,4219% | 1,0000 | 0,9111 | 0,9999 |

### 3. Coerência Operacional e Filtros de Chão de Fábrica
Para blindar os modelos preditivos contra ruídos decorrentes de manobras de campo e inconsistências físicas, a base de dados passou por regras de validação baseadas no comportamento hidráulico real:
* **Inércia de Bomba Desligada:** 17.511 registros (40,53% do tempo) apresentam a velocidade do inversor e a vazão de entrada completamente zeradas. Nesses períodos, qualquer subida abrupta indevida de nível foi tratada para evitar falsas correlações.
* **Vazão de Saída Nula:** Em 1.635 registros (3,78% do tempo), a válvula de saída esteve totalmente fechada. Quedas indevidas de nível nesse cenário foram filtradas e estabilizadas.
* **Planta em Repouso Absoluto:** Foram validados 1.134 registros (2,62% do tempo) com todas as variáveis de fluxo e potência simultaneamente em zero.
* **Eliminação de Manobras Manuais:** Durante a análise de coerência, foram detectadas e eliminadas com sucesso exatamente **600 amostras** associadas a manobras manuais de drenagem ou intervenções externas na planta, estabilizando o horizonte futuro para a IA.

### 4. Força de Acoplamento Linear e Heurística Não-Linear
O acoplamento matemático entre as variáveis preditoras e o alvo preditivo (**Nível Futuro em T + 5 min**) foi mensurado por duas abordagens concorrentes:

* **Associação Linear (Correlação de Pearson):**
  * *Nível Atual:* +0,8616 (Altíssimo acoplamento dinâmico positivo)
  * *Vazão Entrada:* +0,3236
  * *Velocidade Inversor Entrada:* +0,3099
  * *Vazão Saída:* +0,1870
* **Importância Não-Linear (Heurística de Árvores do Random Forest):**
  * *Nível Atual:* Peso de 86,58% no aprendizado
  * *Vazão Entrada:* Peso de 9,56% no aprendizado
  * *Vazão Saída:* Peso de 2,49% no aprendizado
  * *Velocidade Inversor Entrada:* Peso de 1,37% no aprendizado

---

## 🏆 Desempenho dos Modelos e Rigor Estatístico

A validação técnica e a auditoria dos modelos preditivos foram realizadas de forma cronológica rígida, utilizando as amostras finais segregadas como dados inéditos de teste (8.535 linhas). 

As métricas são complementares: o **MAE** aponta o desvio linear típico do dia a dia, o **RMSE** penaliza grandes desvios transientes (garantindo a segurança operacional contra transbordamentos) e o **R² Score** mede o ajuste geométrico das curvas de predição em relação à realidade.

### 🌲 1. Random Forest Regressor
O comitê composto por 100 árvores de decisão processou os nós no processador com altíssima eficiência, alcançando uma estabilidade geométrica impressionante:
* **MAE (Erro Linear Médio):** 0,2937 %
* **RMSE (Risco de Planta/Segurança):** 1,2178 %
* **R² Score (Aderência Global):** 0,9526 (95,26% de precisão explicada)

### 🧠 2. Rede Neural Recorrente LSTM (Deep Learning)
A arquitetura balanceada com 32 neurônios e camadas de Dropout, treinada com memória hidráulica de longo prazo (30 passos), demonstrou excelente suavização temporal:
* **MAE (Erro Médio Absoluto):** 0,7957 %
* **RMSE (Risco Industrial):** 2,8284 %
* **R² Score (Aderência Global):** 0,7368 (73,68% de precisão explicada)

---

## 🎓 Finalidade Acadêmica e Créditos
Este projeto foi desenvolvido como artefato prático-tecnológico para a defesa pública do Trabalho na Matéria de Inteligência Artificial do Instituto Federal de São Paulo (IFSP).

A aplicação simula com sucesso uma arquitetura em ambiente computacional otimizado para CPU, ideal para demonstrações em tempo real de integração entre Engenharia de Dados e Inteligência Artificial Aplicada ao Chão de Fábrica.
