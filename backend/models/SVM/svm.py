import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.model_scripts import prepare_model_data, save_model


def svm_classify(df_music: pd.DataFrame, category:str, verbose=0):
    
    
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    
    svm = SVC(kernel='rbf', C=2, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train, y_train)

    
    y_pred = svm.predict(X_test)
    
    y_prob = svm.predict_proba(X_test)

    save_model(model=svm, mode_name='svm', category=category, scaler=scaler)

    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        
    
        print("\nPredicted Probabilities for the Positive Class (Rock):")
        print(y_prob)

    return svm, y_prob, y_test


if __name__ == "__main__":
    df = pd.read_csv('../../data/processed/music_features.csv')
    accuracy, probabilities = svm_classify(df, "rock")
