# ​Aplicação e Avaliação de Modelos Supervisionados para Classificação da Qualidade do Vinho

## Descrição do Projeto
Este projeto consiste em um trabalho acadêmico que implementa e compara diferentes modelos de classificação para prever se um vinho é de alta qualidade (Premium) ou comum, com base em suas características físico-químicas. Desenvolvido para o Homework 3 da disciplina de Inteligência Computacional Aplicada, o estudo foca na comparação entre modelos lineares e não lineares, utilizando algoritmos como Regressão Logística, LDA, SVM, k-NN e Redes Neurais (MLP).

## Dataset
- Conjunto de Dados: Wine Quality
- Fonte: UCI Machine Learning Repository
- Amostras: 6.497 observações (1.599 vinhos tintos + 4.898 vinhos brancos)
- Características: 11 atributos físico-químicos quantitativos
- Variável alvo: Classe Binária (0: Qualidade < 7; 1: Qualidade ≥ 7) - **Problema de Classificação**
- Divisão: 75% treino (4.872) / 25% teste (1.625)

## Tecnologias e Dependências
- Python 3.7 ou superior
- Bibliotecas principais:
  - numpy - Cálculos numéricos e álgebra linear
  - numpy - Operações matriciais e cálculos numéricos
  - pandas - Manipulação e limpeza de dados
  - scikit-learn - Modelagem (Logística, LDA, SVM, KNN, MLP), Pré-processamento e Métricas
  - matplotlib e seaborn - Visualização de dados e matrizes de confusão
  - scipy - Transformações estatísticas (Yeo-Johnson)
  - tabulate - Formatação de tabelas no console

## Execução do Código
- Primeiro instale as dependências:
  - pip install numpy scipy matplotlib scikit-learn pandas seaborn tabulate
- Depois execute o script principal:
  - python codigo_hw3.py

- O script realizará automaticamente:
   - Download dos datasets (se não existirem)
   - ​Pré-processamento: Tratamento de assimetria (skewness) com Yeo-Johnson e padronização Z-score.
   - ​Otimização: Ajuste de hiperparâmetros via GridSearchCV e RandomizedSearchCV.
   - ​Avaliação Multicritério: Cálculo de Acurácia, Precisão, Recall, F1-Score e AUC-ROC.
   - ​Análise de Erros: Geração de Matrizes de Confusão para comparação entre erros do Tipo I (Falsos Positivos) e Tipo II (Falsos Negativos).

## Arquivos Gerados pelo Código
- outputs_hw3/: Contém tabelas de comparação de modelos e importância de características em formato CSV.
- figures_hw3/: Contém visualizações gráficas, incluindo:
   - ​Curvas ROC para comparação de modelos.
   - ​Matrizes de Confusão de cada algoritmo.
   - ​Gráficos de importância de atributos.
   - ​Curva de aprendizado da Rede Neural.