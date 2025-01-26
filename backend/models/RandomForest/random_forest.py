import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import load
from scripts.config import DATAFRAME_PATH, PATH_TO_BINARY_MODELS, PATH_TO_LAST_STEP_MODELS
from scripts.extract_features import extract_audio_features


from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results


def show_first_ten_rows(file_path='../../data/processed/music_features.csv'):
    df = pd.read_csv(file_path)
    print(df.columns)
    print(df.head(10))


def random_forest_classify(df_music: pd.DataFrame, category:str, verbose=0):
    # df = df_music.drop(columns = ["title"])

    # df = df[df[category].isin([0, 1])]

    # feature_cols = df_music.select_dtypes(include=[np.number]).columns
    # X = df[feature_cols]
    # y = df[category]

    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)



    # można się pobawić parametrami
    rf = RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            min_samples_split=2,
            n_jobs=-1,
            criterion='gini'
        )

    rf.fit(X_train, y_train)
        
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    save_model(model=rf, mode_name='rf', category=category, scaler=scaler)


    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'rf', category)
        
    
    return y_prob, y_test

if __name__ == "__main__":
    # show_first_ten_rows()
    df = pd.read_csv('/Users/szymon/Documents/projekciki/Music-Genre-Classifier/data/processed/music_features_binary_genres.csv')
    accuracy, probabilities, y_test =random_forest_classify(df, "rock")
    print(accuracy)
    print(y_test)

    model = load(f'{PATH_TO_BINARY_MODELS}/rf/model_rock.joblib')
    song_path = "/Users/szymon/Documents/projekciki/Music-Genre-Classifier/Guns N' Roses - Sweet Child O' Mine (Official Music Video).mp3"

    scaler = load(f'{PATH_TO_BINARY_MODELS}/rf/scaler_rock.joblib')
    
    features = np.array([float(num) for num in extract_audio_features(song_path).values()]).reshape(1, -1)

    # Transform features
    X_scaled = scaler.transform(features)
    # X_pca = pca.transform(X_scaled)
    
    # biorę PPB na 1 z każdego modelu, w kolejnych kategoriach
    probs = model.predict_proba(features)[:, 1][0]
    print("probs", probs)
    
