from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results



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



    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'lr', category)

    return y_prob, y_test

if __name__ == "__main__":
    df = pd.read_csv('/Users/szymon/Documents/projekciki/Music-Genre-Classifier/data/processed/music_features_binary_genres.csv')
    accuracy, probabilities, y_test = logistic_regression_classifier(df, "rock")
    print(accuracy)
    print(y_test)
