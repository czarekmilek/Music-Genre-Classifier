from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify
import pandas as pd
import sys, os
import matplotlib.pyplot as plt


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    df = pd.read_csv('data/processed/music_features.csv')

    knn_acc, knn_f1, knn_precision, knn_recall = knn_classify(df)
    svm_acc, svm_f1, svm_precision, svm_recall = svm_classify(df)
    rf_acc, rf_f1, rf_precision, rf_recall = random_forest_classify(df)
    lr_acc, lr_f1, lr_precision, lr_recall = logistic_regression_classifier(df)
    nb_acc, nb_f1, nb_precision, nb_recall = naive_bayes_classify(df)

    classifiers = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
    accuracy_scores = [knn_acc, svm_acc, rf_acc, lr_acc, nb_acc]
    f1_scores = [knn_f1, svm_f1, rf_f1, lr_f1, nb_f1]
    precision_scores = [knn_precision, svm_precision, rf_precision, lr_precision, nb_precision]
    recall_scores = [knn_recall, svm_recall, rf_recall, lr_recall, nb_recall]

    metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    scores = [accuracy_scores, f1_scores, precision_scores, recall_scores]

    for metric, score in zip(metrics, scores):
        print(score)
        plt.figure(figsize=(8, 6))
        plt.bar(classifiers, score, alpha=0.7, color='skyblue')
        plt.ylim(0, 1)
        plt.title(f'{metric} of Different Classifiers', fontsize=16)
        plt.xlabel('Classifier', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
