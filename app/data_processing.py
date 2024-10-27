# data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # Carrega o dataset a partir do caminho fornecido
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Normaliza as features, excluindo a coluna 'Class' (variável alvo)
    scaler = StandardScaler()
    X = data.drop(['Class'], axis=1)
    X_scaled = scaler.fit_transform(X)
    
    # Separa a coluna alvo 'Class' para retornar junto com as features normalizadas
    y = data['Class'].values
    return X_scaled, y

def balance_data(X, y):
    # Converte X para DataFrame e adiciona a coluna 'Class' para balanceamento
    data = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, X.shape[1] + 1)])
    data['Class'] = y

    # Separa classes majoritária e minoritária
    majority = data[data['Class'] == 0]
    minority = data[data['Class'] == 1]

    # Aumenta a classe minoritária para balancear o conjunto de dados
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced_data = pd.concat([majority, minority_upsampled])
    
    # Retorna X e y balanceados
    X_balanced = balanced_data.drop('Class', axis=1).values
    y_balanced = balanced_data['Class'].values
    return X_balanced, y_balanced

def split_data(X, y, test_size=0.2):
    # Aplica o balanceamento antes de dividir em treino e teste
    X_balanced, y_balanced = balance_data(X, y)
    
    # Divide o conjunto balanceado em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test
