# Readme para o Código de Classificação de Dados Meteorológicos

Este código em Python utiliza técnicas de aprendizado de máquina para analisar e prever dados meteorológicos, com foco na previsão de chuva na Austrália. A seguir, apresentamos uma descrição detalhada das principais etapas e componentes do código.

## Dependências

O código depende das seguintes bibliotecas:

- `pandas`: Para manipulação de dados.
- `scikit-learn`: Para algoritmos de aprendizado de máquina e métricas de avaliação.
- `matplotlib`: Para visualização de dados.
- `scipy`: Para testes estatísticos.

## Estrutura do Código

1. **Carregamento dos Dados**:
   - Os dados são carregados a partir de um arquivo CSV (`weatherAUS.csv`), que contém informações meteorológicas.

2. **Preparação dos Dados**:
   - Os dados são agrupados por duas colunas: `RainToday` e `RainTomorrow`.
   - Uma amostra aleatória dos dados é criada para garantir que o conjunto de dados seja embaralhado antes do treinamento.

3. **Divisão do Conjunto de Dados**:
   - Os dados são divididos em conjuntos de treinamento e teste (75% para treinamento e 25% para teste), e o conjunto de treinamento é dividido novamente para validação.

4. **Normalização**:
   - Os dados de entrada são normalizados usando `StandardScaler` para melhorar a performance dos modelos.

5. **Modelos de Classificação**:
   - Diversos algoritmos de aprendizado de máquina são utilizados, incluindo:
     - MLP (Perceptron Multicamadas)
     - KNN (K-Nearest Neighbors)
     - DT (Árvore de Decisão)
     - NB (Naive Bayes)
     - SVM (Support Vector Machine)

   - Para cada modelo, a precisão é calculada através da validação cruzada em 20 iterações.

6. **Combinação de Modelos**:
   - Três métodos diferentes são implementados para combinar as previsões dos modelos: Soma, Produto e Contagem de Borda. Esses métodos têm como objetivo melhorar a precisão das previsões finais.

7. **Testes Estatísticos**:
   - O teste de Kruskal e o teste de Mann-Whitney são realizados para comparar as distribuições das precisões dos diferentes modelos e métodos de combinação.

## Resultados

Os resultados obtidos, incluindo as precisões médias de cada modelo e método de combinação, são armazenados em listas. A precisão média de cada modelo é impressa após cada iteração.

## Considerações Finais

O código está projetado para ser facilmente modificável, permitindo a inclusão de novos algoritmos ou métodos de avaliação conforme necessário. A estrutura modular facilita a adição de novos recursos e a realização de experimentos adicionais. É recomendável que o usuário tenha conhecimentos básicos em Python e em aprendizado de máquina para entender e expandir este código.
