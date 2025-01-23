import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATAFRAME_PATH =  os.path.join(PROCESSED_DATA_DIR, 'music_features_binary_genres.csv')

NON_NUMERICAL_COLUMNS = ['title', 'pop', 'blues', 'hip-hop', 'rock', 'classical', 'reggae', 'country', 'metal', 'techno', 'jazz']

PATH_TO_BINARY_MODELS = "binary_models"
PATH_TO_LAST_STEP_MODELS = "last_step_models"
SPLIT_RANDOM_SEED = 42
SPLIT_PERCENTAGE = 0.1