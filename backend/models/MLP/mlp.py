import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from scripts.model_scripts import prepare_model_data, save_model, log_evaluation_model_results

def mlp_classify(df_music: pd.DataFrame, category: str, verbose=0):
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_music, category)

    # Initialize the MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), # Single hidden layer with 100 neurons
        activation='relu',         # Activation function for the hidden layer
        solver='adam',             # Optimizer
        max_iter=500,              
        random_state=42
    )

    # Train the MLP model
    mlp.fit(X_train, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)

    save_model(model=mlp, mode_name='mlp', category=category, scaler=scaler)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    if verbose:
        log_evaluation_model_results(y_test, y_pred, y_prob, 'mlp', category)


    return y_prob, y_test

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('M:/Projects/Music-Genre-Classifier/backend/data/processed/music_features_binary_genres.csv')
    # Test the MLP classifier
    accuracy, probabilities, y_test = mlp_classify(df, "rock", verbose=1)
    print("=======================")
    print("MLPClassifier Accuracy:", accuracy)
    print("True Labels:", y_test[:10])
    print("Predicted Probabilities:", probabilities[:10])
