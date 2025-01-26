import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.config import PATH_TO_BINARY_MODELS, NON_NUMERICAL_COLUMNS, SPLIT_RANDOM_SEED, SPLIT_PERCENTAGE
import os
from joblib import dump
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import logging
from pathlib import Path
from datetime import datetime


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

def log_evaluation_model_results(y_test, y_pred, y_prob, model_name, category):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    current_dir = Path(__file__).parent
    LOG_FOLDER_NAME = current_dir / f"models_logs/{model_name}/{category}"
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)

    # Set up logging
    LOG_FILE_PATH = os.path.join(LOG_FOLDER_NAME, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(message)s')

    # Clear existing handlers
    logger = logging.getLogger(category)
    logger.handlers = []
    handler = logging.FileHandler(LOG_FILE_PATH)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


    # Log metrics with category logger
    logger.info("Y pred: %s", y_pred)
    logger.info("Y test: %s", y_test)
    logger.info(f"\nClassification Report For {model_name}:")
    logger.info(classification_report(y_test, y_pred))
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info("\nFirst 10 Predicted Probabilities for Class 1:")
    logger.info(y_prob[:10])