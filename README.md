# Detecção de Anomalias em Transações Financeiras

Este projeto implementa um sistema de detecção de anomalias em transações financeiras utilizando autoencoders em Python. A aplicação foi desenvolvida para identificar fraudes em cartões de crédito, priorizando a detecção correta de anomalias.

## Tecnologias Utilizadas

- **Python**: 3.11.9
- **Bibliotecas**: Todas as bibliotecas necessárias estão listadas no arquivo `requirements.txt`.
- **API Kaggle**: Utilizada para baixar o conjunto de dados de fraudes.

## Funcionalidades

- Download e pré-processamento de dados.
- Treinamento de um modelo de autoencoder.
- Detecção de anomalias em transações financeiras.
- Avaliação do desempenho do modelo com métricas como recall, precisão e acurácia.

## Como Executar

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DA_PASTA>

2. Crie uma virtual environment:
   ```bash
   python -m venv .venv

3. Ative a virtual environment:
    ° No Windows:
    ```bash
    venv\Scripts\activate
    ° No macOS e Linux:
    ```bash
    source venv/bin/activate

4. Instale as dependências:
    ```bash
    pip install -r requirements.txt

5. Execute a aplicação:
    ```bash
    python run.py




