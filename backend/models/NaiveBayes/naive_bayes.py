from sklearn.naive_bayes import GaussianNB
import pandas as pd

from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results


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


    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'nb', category)
       
    
    return y_prob, y_test

if __name__ == "__main__":
    df = pd.read_csv('../../data/processed/music_features.csv')
    accuracy, probabilities = naive_bayes_classify(df, "rock")