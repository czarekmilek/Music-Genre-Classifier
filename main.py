from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify
import pandas as pd
import sys, os
import matplotlib.pyplot as plt



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    df = pd.read_csv('data/processed/music_features.csv')
    blockPrint()
    knn = knn_classify(df)
    svm = svm_classify(df)
    random_forest = random_forest_classify(df)
    logistic_regression = logistic_regression_classifier(df)
    naive_bayes = naive_bayes_classify(df)
    enablePrint()



    # Labels and scores
    classifiers = ['KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
    scores = [knn, svm, random_forest, logistic_regression, naive_bayes]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, scores, alpha=0.7)
    plt.ylim(0, 1)
    plt.title('Accuracy of Different Classifiers', fontsize=16)
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()