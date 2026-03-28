import sys
from src.logger import logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer

class training_pipeline:

    def start_training(self):
        try:
            logger.info("Training pipeline started")

            ingestion = DataIngestion()
            train_path,test_path = ingestion.inititate_data_ingestion()

            transformation = DataTransformation()
            X_train,X_test,y_train,y_test = transformation.initiate_data_transformation(train_path,test_path)

            trainer = Model_trainer()
            accuracy = trainer.initiate_model_trainer(X_train,X_test,y_train,y_test)

            logger.info("Training pipeline completed")
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)
