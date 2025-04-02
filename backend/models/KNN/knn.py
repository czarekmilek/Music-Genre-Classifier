from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results


def knn_classify(df_music: pd.DataFrame, category:str,  n_neighbors=7, verbose=0):


    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    # print("probabilities: ", y_prob)
    # print("label: ", y_test)
    

    save_model(model=knn, mode_name='knn', category=category, scaler=scaler)


    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'knn', category)


    return y_prob, y_test

if __name__ == "__main__":
    # run as module  python -m models.KNN.knn
    # Load the dataset
    df = pd.read_csv('/Users/szymon/Documents/projekciki/Music-Genre-Classifier/backend/data/processed/music_features_binary_genres.csv')
    # Run the KNN classifier
    probabilities, y = knn_classify(df,"rock", verbose=1)
    print("kkn probabilities for each song from the test_set being rock:", probabilities[0])
    print("true label for the probability from above", y[1])
