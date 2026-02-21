import os
import sys

import numpy as np 
import pandas as pd
from src.logger import logging
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,model_params):
    try:
        report:dict={}
        for model_name in models.keys():
            param=model_params[model_name]
            model=models[model_name]
            grid=GridSearchCV(model,param_grid=param,cv=3,n_jobs=6,scoring="r2")
            grid.fit(x_train,y_train)

            model.set_params(**grid.best_params_)
            model.fit(x_train,y_train)

            y_pred=model.predict(x_test)
            model_test_score=r2_score(y_test,y_pred)
            report[model_name]=model_test_score

        logging.info("Training Of Models Completed")

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as fileObj:
            return dill.load(fileObj)

    except Exception as e:
        raise CustomException(e,sys)