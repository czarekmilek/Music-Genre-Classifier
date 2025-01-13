import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import pandas as pd

def show_first_ten_rows(file_path='music_features.csv'):
    df = pd.read_csv(file_path)
    print(df.columns)
    print(df.head(10))


def random_forest_classify(df_music):
    df = df_music.drop(columns = ["title"])
    feature_cols = df_music.select_dtypes(include=[np.number]).columns
    X = df_music[feature_cols]
    
    # nadajemy numery kategoriom po prostu
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_music['category'])


    # dzielimy z zachowaniem proporcji
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            random_state=42, 
            test_size=0.2,
            stratify=y
        )


    # można się pobawić parametrami
    rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            min_samples_split=10,
            n_jobs=-1,
            criterion='gini'
        )

    rf.fit(X_train, y_train)
        
    y_pred = rf.predict(X_test)
    score = rf.score(X_test, y_test)

    print(classification_report(y_test, y_pred))
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {score:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    })
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))
    
    return score

if __name__ == "__main__":
    # show_first_ten_rows()
    df = pd.read_csv('music_features.csv')
    random_forest_classify(df)
