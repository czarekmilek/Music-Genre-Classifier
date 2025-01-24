import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

from scripts.model_scripts import prepare_model_data, save_model


def knn_classify(df_music: pd.DataFrame, category:str,  n_neighbors=7, verbose=0):


    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    # print("probabilities: ", y_prob[1])
    # print("label: ", y_pred[1])
    

    save_model(model=knn, mode_name='knn', category=category, scaler=scaler)


    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

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
