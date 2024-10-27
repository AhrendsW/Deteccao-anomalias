# anomaly_detection.py

from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

def build_autoencoder(input_dim):
    """Constrói um modelo de autoencoder."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(14, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

def train_autoencoder(X_train, epochs=50, batch_size=256):
    """Treina o autoencoder."""
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)

    print("Treinando o autoencoder...")
    autoencoder.fit(X_train, X_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)
    
    return autoencoder

def calculate_threshold(errors, quantile=0.99):
    # Usa quantil para determinar threshold
    return np.quantile(errors, quantile)

def detect_anomalies(model, X_test, threshold=None):
    # Obtém os erros de reconstrução
    reconstructions = model.predict(X_test)
    errors = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    
    # Calcula o threshold se não estiver especificado
    threshold = threshold or calculate_threshold(errors)
    
    # Marca como anomalia os erros acima do threshold
    return np.array([1 if error > threshold else 0 for error in errors]), threshold


def evaluate_model(y_true, y_pred):
    """Calcula as métricas de desempenho do modelo com foco em recall e precisão."""
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    return recall, precision, accuracy

def cross_validate(X, y, k_folds=5):
    """Aplica validação cruzada para avaliar o modelo e retorna as métricas médias."""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    recalls, precisions, accuracies = [], [], []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        autoencoder = train_autoencoder(X_train)
        anomalies, _ = detect_anomalies(autoencoder, X_test)
        
        # Avalia o modelo e armazena as métricas
        recall, precision, accuracy = evaluate_model(y_test, anomalies)
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)

    return np.mean(recalls), np.mean(precisions), np.mean(accuracies)

def detect_anomalies_with_threshold(autoencoder, X_test, y_test):
    """Detecta anomalias e avalia o modelo com diferentes limiares."""
    _, reconstruction_error = detect_anomalies(autoencoder, X_test)

    # Testa diferentes limiares para maximizar o recall
    best_recall = 0
    best_threshold = 0

    for threshold in np.arange(0.01, 1.0, 0.01):
        anomalies = reconstruction_error > threshold
        recall = recall_score(y_test, anomalies)

        if recall > best_recall:
            best_recall = recall
            best_threshold = threshold

    print(f"Melhor limiar encontrado: {best_threshold:.2f} com recall: {best_recall:.4f}")
    return best_threshold
