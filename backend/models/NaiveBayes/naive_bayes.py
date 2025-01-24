import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.model_scripts import prepare_model_data, save_model


def naive_bayes_classify(df_music:pd.DataFrame, category:str, verbose=0):
    # df = df_music.drop(columns=["title"])
    # df = df[df[category].isin([0, 1])]

    # feature_cols = df.select_dtypes(include=[np.number]).columns
    # X = df[feature_cols]
    # y = df[category]

    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)


    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)


    save_model(model=nb, mode_name='nb', category=category, scaler=scaler)


    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return accuracy, y_prob, y_test

if __name__ == "__main__":
    df = pd.read_csv('../../data/processed/music_features.csv')
    accuracy, probabilities = naive_bayes_classify(df, "rock")