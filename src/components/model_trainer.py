import os 
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import save_object

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    Trained_Model_path=os.path.join('artifacts','model.pickle')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=[
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ]
            models:dict = {
                "Random Forest": RandomForestRegressor(),
                "KNN":KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_params:dict ={
                "Random Forest":{
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]},
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]},
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 15]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0]},
                "Linear Regression": {
                    "fit_intercept": [True, False]},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7]},
                "CatBoosting Regressor": {
                    "depth": [4, 6, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200]},
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [1.0, 0.5, 0.1],
                    "loss": ["linear", "square", "exponential"]}
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,model_params=model_params)
            logging.info("Model Training Completed")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            predicted=best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)


            save_object(file_path=self.model_trainer_config.Trained_Model_path,
                        obj=best_model)
            logging.info("Best Model Saved")
            
            return (best_model_name,r2_square)
        
        except Exception as e:
            raise CustomException(e,sys)
        