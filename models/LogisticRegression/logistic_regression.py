from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from scripts.model_scripts import prepare_model_data, save_model


def logistic_regression_classifier(df_music: pd.DataFrame, category: str, verbose=0):
    
    # df = df_music.drop(columns=["title"])
    # df = df[df[category].isin([0, 1])]
    # feature_cols = df.select_dtypes(include=[np.number]).columns

    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    lr = LogisticRegression(
        random_state=42, 
        max_iter=1000,  
    )
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)

    save_model(model=lr, mode_name='lr', category=category, scaler=scaler)


    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print("\nPredicted Probabilities for Class 1:")
        print(y_prob[:10])  

    return accuracy, y_prob, y_test

if __name__ == "__main__":
    df = pd.read_csv('/Users/szymon/Documents/projekciki/Music-Genre-Classifier/data/processed/music_features_binary_genres.csv')
    accuracy, probabilities, y_test = logistic_regression_classifier(df, "rock")
    print(accuracy)
    print(y_test)
