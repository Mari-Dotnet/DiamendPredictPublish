from sklearn.impute import SimpleImputer ## handlin missing value
from sklearn.preprocessing import StandardScaler ## feature scalling
from sklearn.preprocessing import OrdinalEncoder ## ordinal encoding

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os,sys
from src.logger import logging 
from src.exception import CustomException
import pandas as pd
import numpy as np

from src.utils import save_object

## Data transfer config
from dataclasses import dataclass

@dataclass
class DataTranfermationConfig:
    preprocessor_obj_filepath=os.path.join('artifacts','preprocessor.pkl')

class DataTranfermation:
    def __init__(self) :
        self.datatranferation_config=DataTranfermationConfig()

    def get_datatranfrmaion_object(self):
        try:
            logging.info("data tranfermation started")
             # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline started")

            #numericalpipeline
            num_pipeline=Pipeline(
                steps=[
                ('SimpleImputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            ##categtorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('SimpleImputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )


            ##pre-processing started
            pre_processor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info("pipeline completed")

            return pre_processor
        except Exception as e:
            logging.info("error occoured in datatranfermation")
        
    def initiate_data_tranfermation(self,train_path,test_path):
        logging.info('initiate data tranfermation started')
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(" read data from train and test data path")
            logging.info(f"train data shape{train_df.shape}, test data shape{test_df.shape}")

            pre_processor_obj=self.get_datatranfrmaion_object()

            target_name='price'
            dropcolumn=[target_name,'id']


            ##split indepndent and dependent feature
            input_featue_train_df=train_df.drop(columns=dropcolumn)
            target_feature_trian_df=train_df[target_name]

            input_featue_test_df=test_df.drop(columns=dropcolumn)
            target_feature_test_df=test_df[target_name]
            

            ## Prepocessing started
            logging.info('proprocesing started')
            input_feature_traindf=pre_processor_obj.fit_transform(input_featue_train_df)
            input_feature_test=pre_processor_obj.transform(input_featue_test_df)

            logging.info('preprocessing completed')

            train_arr=np.c_[input_feature_traindf,np.array(target_feature_trian_df)]
            test_arr=np.c_[input_feature_test,np.array(target_feature_test_df)]

            save_object(
                file_path=self.datatranferation_config.preprocessor_obj_filepath,
                obj=pre_processor_obj
            )

            logging.info('pickle file saved')
            return(
                train_arr,
                test_arr,
                self.datatranferation_config.preprocessor_obj_filepath
            )
        except Exception as e:
            raise CustomException(e,sys)

