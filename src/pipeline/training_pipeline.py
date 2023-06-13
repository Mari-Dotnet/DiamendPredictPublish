import os,sys

from src.logger import logging
from src.exception import CustomException
import pandas as pd

from  src.components.data_injection import DataIngestion

from src.components.data_transformation import DataTranfermation

from src.components.model_trainer import Model_Training


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_tanfermation=DataTranfermation()
    trian_arr,test_arr,pikle_path=data_tanfermation.initiate_data_tranfermation(train_data_path,test_data_path)

    print(pikle_path)

    model_trainer=Model_Training()
    model_trainer.Model_traininer_initiator(trian_arr,test_arr)