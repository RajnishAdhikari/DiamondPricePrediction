# here we will all the code or functionality which will be common to all the project
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train,Y_train, X_test,Y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            #Train model
            model.fit(X_train,Y_train)

            #Predict Testing data
            Y_test_pred = model.predict(X_test)

            #Get R2 scores for train and test data
            test_model_score = r2_score(Y_test, Y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e,sys)
