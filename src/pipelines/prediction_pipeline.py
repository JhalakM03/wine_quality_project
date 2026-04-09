import sys
import pandas as pd
from src.exception import CustomException
from utils import load_object

class PredictionPipeline:

    def __init__(self):
        self.model_path = 'artifact/model.pkl'
        self.preprocessor_path = 'artifact/preprocessor.pkl'

    def predict(self, features):

        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)

            predictions = preprocessor.predict(data_scaled)

            return predictions
        
        except Exception as e:
             raise CustomException(e,sys)
        

class CustomData:

    def __init__(
            self,
            fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
    ):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                "fixed acidity": [self.fixed_acidity],
                "volatile acidity": [self.volatile_acidity],
                "citric acid": [self.citric_acid],
                "residual sugar": [self.residual_sugar],
                "chlorides": [self.chlorides],
                "free sulfur dioxide": [self.free_sulfur_dioxide],
                "total sulfur dioxide": [self.total_sulfur_dioxide],
                "density": [self.density],
                "pH": [self.pH],
                "sulphates": [self.sulphates],
                "alcohol": [self.alcohol]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)