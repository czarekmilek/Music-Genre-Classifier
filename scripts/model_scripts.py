import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.config import PATH_TO_BINARY_MODELS, NON_NUMERICAL_COLUMNS, SPLIT_RANDOM_SEED, SPLIT_PERCENTAGE
import os
from joblib import dump


def prepare_model_data(df_music: pd.DataFrame, category:str):

    df = df_music[df_music[category].isin([0, 1])]


    feature_cols = [col for col in df.columns if col not in NON_NUMERICAL_COLUMNS]
    # print(feature_cols)
    # load data from the table
    X = df[feature_cols].values
    # print(X)

    y = df[category]

    # print(X.shape)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # # Apply PCA for dimensionality reduction
    # pca = PCA(n_components=36)
    # X_pca = pca.fit_transform(X_scaled)


    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, random_state=SPLIT_RANDOM_SEED, test_size=SPLIT_PERCENTAGE, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def save_model(model, mode_name: str, category: str, scaler):
    os.makedirs(f'{PATH_TO_BINARY_MODELS}/{mode_name}', exist_ok=True)
    dump(model, f'{PATH_TO_BINARY_MODELS}/{mode_name}/model_{category}.joblib')
    dump(scaler, f'{PATH_TO_BINARY_MODELS}/{mode_name}/scaler_{category}.joblib')