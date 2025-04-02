from typing import List, Callable, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
import logging
from sklearn.metrics import log_loss
from datetime import datetime
from pathlib import Path

# RUN AS MODULE python -m predict.logictic_regression_predict


def train_logistic_regression(X, y, category, model_names, verbose=0):
    
    # print(self.X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.2, stratify=y
    )


    model = LogisticRegression(
        # class_weight='balanced',
        random_state=42,
        max_iter=1000, 
        solver='lbfgs',
        tol=1e-4
        # penalty='l2' 
    )
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # loss after training on the test set
    ce_loss = log_loss(y_test, y_prob)

    os.makedirs(f'{PATH_TO_LAST_STEP_MODELS}/categorized_regression', exist_ok=True)
    dump(model, f'{PATH_TO_LAST_STEP_MODELS}/categorized_regression/{category}.joblib')

    if verbose:
        coef_importance = pd.DataFrame({
            'feature': [model_names[i] for i in range(len(model.coef_[0]))],
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')

        current_dir = Path(__file__).parent
        LOG_FOLDER_NAME = current_dir / f"logistic_regression_log/{category}"
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



        # # Log metrics with category logger
        # logger.info("Y pred: %s", y_pred)
        # logger.info("Y test: %s", y_test)
        # logger.info("\nFeature Importance by Coefficients:")
        # logger.info(coef_importance)
        # logger.info("Cross Entropy Loss on training set: %f", ce_loss)
        # logger.info("\nClassification Report For Logistic Regression:")
        # logger.info(classification_report(y_test, y_pred))
        # logger.info(f"Accuracy: {accuracy:.4f}")
        # logger.info(f"F1 Score: {f1:.4f}")
        # logger.info(f"Recall: {recall:.4f}")
        # logger.info(f"Precision: {precision:.4f}")
        # logger.info("\nFirst 10 Predicted Probabilities for Class 1:")
        # logger.info(y_prob[:10])

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

    # train_logistic_regression(verbose=0)
