import numpy as np

def split_features_labels(df_music, genre_count=10):

    # Drop title column if exists
    df = df_music.drop(columns=["title"], errors="ignore")

    # First n (n = len(df.columns) - genre_count) represent audio file features
    feature_cols = df.columns[:-genre_count] 
    df_features = df[feature_cols]

    # Ensure that only numeric data is selected
    feature_cols_num = df_features.select_dtypes(include=[np.number]).columns
    X = df_features[feature_cols_num]

    # Last n (n=genre_count) columns represent music genres
    genre_cols = df.columns[-genre_count:] 

    # For each row, create a set of genre names where the corresponding cell value is 1 
    y = df[genre_cols].apply(lambda row: {col for col in genre_cols if row[col] == 1}, axis=1)

    return X, y
