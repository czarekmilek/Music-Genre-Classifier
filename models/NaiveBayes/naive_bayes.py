import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def naive_bayes_classify(df_music):
    df = df_music.drop(columns=["title"])
    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
    

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_music['category'])

    # Scaler - standardizes each column to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=36) 
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=42, test_size=0.2, stratify=y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')


    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return accuracy, f1, precision, recall

if __name__ == "__main__":
    df = pd.read_csv('../../data/processed/music_features.csv')
    model, label_encoder = naive_bayes_classify(df)