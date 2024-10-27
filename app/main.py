# main.py

import os
import pandas as pd
import kaggle
from .data_processing import load_data, preprocess_data, split_data
from .anomaly_detection import train_autoencoder, detect_anomalies, evaluate_model, cross_validate

def download_data():
    """Verifica se o arquivo dos dados já está disponível e, se não, faz o download."""
    raw_data_path = "data/raw/creditcard.csv"  # Caminho onde os dados serão armazenados
    
    # Verifica se o arquivo já existe
    if not os.path.isfile(raw_data_path):
        print("Baixando o dataset... Isso pode levar alguns minutos.")
        kaggle.api.dataset_download_files("mlg-ulb/creditcardfraud", path="data/raw", unzip=True)
        print("Dataset baixado e extraído com sucesso!")
    else:
        print("O dataset já está disponível. Vamos usar ele!")

    return raw_data_path

def run_app():
    """Função principal para rodar a aplicação de detecção de anomalias."""
    print("Iniciando a aplicação de detecção de anomalias...")
    
    # Faz o download dos dados (se necessário) e obtém o caminho do arquivo
    raw_data_path = download_data()
    
    # Carrega os dados do arquivo CSV
    df = load_data(raw_data_path)
    
    # Prepara os dados para o modelo
    X, y = preprocess_data(df)
    
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Dados pré-processados e divididos com sucesso!")
    print(f"Tamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de teste: {X_test.shape}")

    # Treina o autoencoder com o conjunto original de dados
    autoencoder = train_autoencoder(X_train)

    # Detecta anomalias e calcula métricas no conjunto de teste
    anomalies, reconstruction_error = detect_anomalies(autoencoder, X_test)
    recall, precision, accuracy = evaluate_model(y_test, anomalies)

    print("\nMétricas antes da validação cruzada:")
    print(f"Recall: {recall:.4f}, Precisão: {precision:.4f}, Acurácia: {accuracy:.4f}")

    # Avaliação com validação cruzada
    print("\nIniciando a validação cruzada...")
    mean_recall, mean_precision, mean_accuracy = cross_validate(X, y)
    
    print("\nMétricas após a validação cruzada:")
    print(f"Recall Médio: {mean_recall:.4f}, Precisão Média: {mean_precision:.4f}, Acurácia Média: {mean_accuracy:.4f}")

if __name__ == "__main__":
    run_app()
