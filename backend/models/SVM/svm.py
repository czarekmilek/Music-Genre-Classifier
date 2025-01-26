import numpy as np
import pandas as pd
from sklearn.svm import SVC

from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results


def svm_classify(df_music: pd.DataFrame, category:str, verbose=0):
    
    
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    
    svm = SVC(kernel='rbf', C=2, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train, y_train)

    
    y_pred = svm.predict(X_test)
    
    y_prob = svm.predict_proba(X_test)

    save_model(model=svm, mode_name='svm', category=category, scaler=scaler)

    

    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'svm', category)

    return y_prob, y_test


if __name__ == "__main__":
    df = pd.read_csv('../../data/processed/music_features.csv')
    accuracy, probabilities = svm_classify(df, "rock")
