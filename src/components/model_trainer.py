import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.utils import evaluate_model,save_object

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trainer_model_tainer_path=os.path.join('artifacts','model.pkl')

class Model_Training:
    def __init__(self):
        self.model_tainer_config=ModelTrainerConfig()

    def Model_traininer_initiator(self,train_array,test_array):
        try:
            logging.info("model training started split the train and test values")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor(),
            'RandomForestRegressor':RandomForestRegressor()
            }

            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)

            #get the best model score
            best_score=max(model_report.values())

            ##to get the best model
            best_model=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]


            print(f"best model found, model name {best_model} and r2 score {best_score}")
            logging.info(f"best model found, model name {best_model} and r2 score {best_score}")

            best_model_obj = models[best_model]
            save_object(
                file_path=self.model_tainer_config.trainer_model_tainer_path,
                obj=best_model_obj
                )
        except Exception as e:
            raise CustomException(e,sys)