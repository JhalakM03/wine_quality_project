import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.logger import logger
from utils import save_object
from src.exception import CustomException
from src.config import DataTransformationConfig

class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig

    def initiate_data_transformation(self,train_path,test_path):

        try:
            logger.info('Data Transformation Started')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if "Id" in train_df.columns:
                train_df.drop("Id", axis=1, inplace=True)
                test_df.drop("Id", axis=1, inplace=True)

            train_df['quality_label']=train_df['quality'].apply(lambda x: 1 if x>=6 else 0)
            test_df['quality_label']=test_df['quality'].apply(lambda x: 1 if x>=6 else 0)

            X_train = train_df.drop(['quality','quality_label'], axis=1)
            y_train = train_df['quality_label']

            X_test = test_df.drop(['quality', 'quality_label'], axis=1)
            y_test = test_df['quality_label']

            pipeline=Pipeline([
                ('scaler', StandardScaler())
            ])

            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)

            save_object(
                self.transformation_config.preprocessor_obj_file_path,pipeline
            )

            logger.info('Preprocessor Saved')

            return X_train_transformed, X_test_transformed, y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)
        