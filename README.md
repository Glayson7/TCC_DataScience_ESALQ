### README para o Modelo de Classificação de Criticidade de Defeitos

#### Resumo

Este projeto tem como objetivo desenvolver um modelo de Machine Learning para classificar a criticidade de defeitos relatados. Utilizamos o XGBoost como algoritmo de classificação e aplicamos várias técnicas de pré-processamento e balanceamento de dados, como Label Encoding e SMOTE.

#### Pré-requisitos

- Python 3.x
- Bibliotecas listadas em `requisitos.txt`

#### Como Instalar

1. Clone o repositório:
    ```
    git clone https://github.com/Glayson7/TCC_DataScience_ESALQ.git
    ```

2. Instale as dependências:
    ```
    pip install -r requisitos.txt
    ```

#### Estrutura do Projeto

- `TCC_XGBoost.py`: Contém o código fonte do projeto.
- `requisitos.txt`: Lista todas as dependências do projeto.
- `Gráficos`: Diretório contendo gráficos gerados durante a análise.

#### Como Executar

1. Navegue até o diretório do projeto:
    ```
    cd TCC_DataScience_ESALQ
    ```

2. Execute o script Python:
    ```
    python TCC_XGBoost.py
    ```

#### Metodologia

1. **Carregamento dos Dados**: Os dados são carregados a partir de um arquivo Excel.
2. **Pré-processamento de Dados**: As variáveis categóricas são codificadas usando Label Encoding.
3. **Análise Exploratória de Dados**: Realizada para entender a distribuição das classes.
4. **Balanceamento de Classes**: Usado SMOTE para balancear as classes.
5. **Separação de Dados**: Os dados são divididos em conjuntos de treinamento e teste.
6. **Otimização de Modelo**: Grid Search com validação cruzada para otimizar os parâmetros do modelo XGBoost.
7. **Avaliação de Modelo**: Métricas como Acurácia, Precisão, Recall e F1-Score são calculadas.

#### Métricas

- **Acurácia**: Mede a porcentagem de classificações corretas do modelo.
- **Precisão**: Avalia a proporção de verdadeiros positivos entre as previsões positivas.
- **Recall**: Avalia a proporção de verdadeiros positivos entre todas as amostras positivas reais.
- **F1-Score**: Média harmônica entre Precisão e Recall.

#### Importância dos Recursos

A importância dos recursos é avaliada e plotada para entender quais variáveis têm mais impacto na classificação.

#### Contato

Para mais informações, sinta-se à vontade para entrar em contato com o autor através do [GitHub](https://github.com/Glayson7).

