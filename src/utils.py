import sys,os
import pickle
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pickle 

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        print("dir_path",dir_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)



def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            ##
            y_pred=model.predict(X_test)
            score=r2_score(y_test,y_pred)
            report[list(models.keys())[i]]=score
        return report
    except Exception as e:
        raise CustomException(e,sys)


def load_object(filepath):
    try:
        with open(filepath,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)