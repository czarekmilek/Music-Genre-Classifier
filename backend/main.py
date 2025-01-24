import numpy as np
from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify
from scripts.extract_features import extract_audio_features
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def predict_genre(file_path, model, label_encoder, scaler, pca, csv_columns):

    try:
        features = extract_audio_features(file_path)
        feature_dict = {col: features.get(col, 0) for col in csv_columns}
        feature_df = pd.DataFrame([feature_dict])
        scaled_features = scaler.transform(feature_df)
        pca_features = pca.transform(scaled_features)
        predicted_label = model.predict(pca_features)
        predicted_genre = label_encoder.inverse_transform(predicted_label)[0]

        return predicted_genre

    except Exception as e:
        print(f"Error predicting genre for {file_path}: {e}")
        return None
    
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    df = pd.read_csv('data/processed/music_features.csv')

    # knn_acc, knn_f1, knn_precision, knn_recall, _, _ = knn_classify(df)
    # svm_acc, svm_f1, svm_precision, svm_recall, _, _ = svm_classify(df)
    # rf_acc, rf_f1, rf_precision, rf_recall, _, _ = random_forest_classify(df)
    # lr_acc, lr_f1, lr_precision, lr_recall, _, _ = logistic_regression_classifier(df)
    # nb_acc, nb_f1, nb_precision, nb_recall, _, _ = naive_bayes_classify(df)

    # classifiers = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
    # accuracy_scores = [knn_acc, svm_acc, rf_acc, lr_acc, nb_acc]
    # f1_scores = [knn_f1, svm_f1, rf_f1, lr_f1, nb_f1]
    # precision_scores = [knn_precision, svm_precision, rf_precision, lr_precision, nb_precision]
    # recall_scores = [knn_recall, svm_recall, rf_recall, lr_recall, nb_recall]

    # metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    # scores = [accuracy_scores, f1_scores, precision_scores, recall_scores]

    # for metric, score in zip(metrics, scores):
    #     print(score)
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(classifiers, score, alpha=0.7, color='skyblue')
    #     plt.ylim(0, 1)
    #     plt.title(f'{metric} of Different Classifiers', fontsize=16)
    #     plt.xlabel('Classifier', fontsize=12)
    #     plt.ylabel(metric, fontsize=12)
    #     plt.grid(axis='y', linestyle='--', alpha=0.6)
    #     plt.tight_layout()
    #     plt.show()



    _, _, _, _, model, label_encoder = knn_classify(df)


    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=36)
    pca.fit(X_scaled)

    csv_columns = feature_cols.tolist()

    new_file_path = "music_samples/rock/Layla.mp3" 
    predicted_genre = predict_genre(new_file_path, model, label_encoder, scaler, pca, csv_columns)
    if predicted_genre:
        print(f"Przewidywany gatunek dla {new_file_path}: {predicted_genre}")