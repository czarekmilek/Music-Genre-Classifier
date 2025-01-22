import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from joblib import dump
import os
from scripts.config import PATH_TO_BINARY_MODELS, NON_NUMERICAL_COLUMNS


def knn_classify(df_music: pd.DataFrame, category:str,  n_neighbors=7, verbose=0):
         
    df = df_music[df_music[category].isin([0, 1])]


    feature_cols = [col for col in df.columns if col not in NON_NUMERICAL_COLUMNS]
    # print(feature_cols)
    # load data from the table
    X = df[feature_cols].values
    # print(X)

    y = df[category]

    print(X.shape)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # # Apply PCA for dimensionality reduction
    # pca = PCA(n_components=36)
    # X_pca = pca.fit_transform(X_scaled)


    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, random_state=42, test_size=0.2, stratify=y
    )

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    # print("probabilities: ", y_prob[1])
    # print("label: ", y_pred[1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    os.makedirs(f'{PATH_TO_BINARY_MODELS}/knn', exist_ok=True)
    dump(knn, f'{PATH_TO_BINARY_MODELS}/knn/knn_model_{category}.joblib')
    dump(scaler, f'{PATH_TO_BINARY_MODELS}/knn/scaler_{category}.joblib')
    # dump(pca, f'{PATH_TO_BINARY_MODELS}/knn/pca_{category}.joblib')
    dump(feature_cols, f'{PATH_TO_BINARY_MODELS}/knn/features_{category}.joblib')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")


    return accuracy, y_prob, y_test

if __name__ == "__main__":
    # run as module  python -m models.KNN.knn
    # Load the dataset
    df = pd.read_csv('/Users/szymon/Documents/projekciki/Music-Genre-Classifier/data/processed/music_features_binary_genres.csv')
    # Run the KNN classifier
    accuracy, probabilities, y = knn_classify(df,"rock", verbose=1)
    print("kkn probabilities for each song from the test_set being rock:", probabilities[0])
    print("true label for the probability from above", y[1])
