import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from xgboost import XGBClassifier
from scripts.model_scripts import prepare_model_data, save_model

def xgboost_classify(df_music: pd.DataFrame, category: str, verbose=0):
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    # Initialize and train the XGBoost classifier
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)

    save_model(model=xgb, mode_name='xgboost', category=category, scaler=scaler)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")

    return y_prob, y_test

if __name__ == "__main__":
    df = pd.read_csv('M:/Projects/Music-Genre-Classifier/backend/data/processed/music_features_binary_genres.csv')
    accuracy, probabilities, y_test = xgboost_classify(df, "pop", verbose=1)

    print("=======================")
    print("Accuracy:", accuracy)
    print("True Labels:", y_test)
