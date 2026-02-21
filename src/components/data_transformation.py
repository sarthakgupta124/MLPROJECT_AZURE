import sys 
import os 

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import numpy as np 
import pandas as pd

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transforamtion_obj(self):
        try:
            num_features=[ 'reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('standardScaler',StandardScaler())
            ])

            cat_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ('columnTransformer',OneHotEncoder(drop='first'))
            ])

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")


            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipelines",cat_pipeline,cat_features)

                ]


            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj=self.get_data_transforamtion_obj()

            target_col_name="math_score"
            input_train=train_df.drop(columns=target_col_name,axis=1)
            target_train=train_df[target_col_name]

            input_test=test_df.drop(columns=target_col_name,axis=1)
            target_test=test_df[target_col_name]


            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_train)
            input_feature_test_arr=preprocessor_obj.transform(input_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_train)]  
            test_arr = np.c_[input_feature_test_arr, np.array(target_test)] ## it recombine the target and input features in single object ,np.c_ it concatinate the data column wise

            logging.info(f"Saved preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)    