import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def knn_classify(df_music: pd.DataFrame, category:str,  n_neighbors=7):
     
    df = df_music.drop(columns=["title"])
    df = df[df[category].isin([0, 1])]

    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=36)
    X_pca = pca.fit_transform(X_scaled)

    y = df[category];
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, random_state=42, test_size=0.2, stratify=y
    )

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy, y_prob

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('../../data/processed/music_features.csv')
    # Run the KNN classifier
    knn_model, probabilities = knn_classify(df,"rock")
