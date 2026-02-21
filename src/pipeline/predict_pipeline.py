from src.utils import load_obj
from src.logger import logging 
from src.exception import CustomException
import sys
import pandas as pd


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_education = parental_level_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_education": [self.parental_level_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model=load_obj("artifacts\model.pickle")
            preprocessor=load_obj("artifacts\proprocessor.pkl")

            scaled_fea=preprocessor.transform(features)

            pred_data=model.predict(scaled_fea)
            return pred_data
        except Exception as e:
            raise CustomException(e,sys)