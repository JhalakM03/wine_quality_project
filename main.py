from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer

if __name__ == "__main__":
    
    #data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.inititate_data_ingestion

    #data transformation
    transformation = DataTransformation()
    X_train,X_test,y_train, y_test = transformation.initiate_data_transformation(train_path,test_path)

    #model training
    model_trainer = Model_trainer()
    accuracy = model_trainer.initiate_model_trainer(X_train,X_test,y_train,y_test)

    print("FInal Model accuracy:", accuracy)