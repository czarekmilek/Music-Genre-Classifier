from typing import List, Callable, Tuple
import pandas as pd
import numpy as np
from models.KNN.knn import knn_classify
from models.NaiveBayes.naive_bayes import naive_bayes_classify
from models.LogisticRegression.logistic_regression import logistic_regression_classifier
from models.RandomForest.random_forest import random_forest_classify
from models.SVM.svm import svm_classify
from scripts.config import DATAFRAME_PATH
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from joblib import dump
import os
from scripts.config import PATH_TO_LAST_STEP_MODELS

# RUN AS MODULE python -m predict.logictic_regression_predict


def train_logistic_regression(X, y, category, verbose=0):
    
    # print(self.X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2
    )


    model = LogisticRegression(
    random_state=42, 
        max_iter=1000,  
    )
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)


    y_prob = model.predict_proba(X_test)[:, 1]

    os.makedirs(f'{PATH_TO_LAST_STEP_MODELS}/categorized_regression', exist_ok=True)
    dump(model, f'{PATH_TO_LAST_STEP_MODELS}/categorized_regression/{category}.joblib')

    if verbose:
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')

        print("Y pred: ", y_pred)
        print("Y test: ", y_test)

        print(f"\nClassification Report For Logistic Regression:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print("\nPredicted Probabilities for Class 1:")
        print(y_prob[:10])  

    return model

            



if __name__ == "__main__":
    pass
    # df = pd.read_csv(DATAFRAME_PATH)
                                                       
    # rock_logistic = MakeLogisticRegression(df, 'rock', [knn_classify,
    #                                                     naive_bayes_classify,
    #                                                     logistic_regression_classifier,
    #                                                     random_forest_classify,
    #                                                     svm_classify
    #                                                       ])

    # rock_logistic.train_logistic_regression(verbose=0)
