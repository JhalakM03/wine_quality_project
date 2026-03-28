import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object
from src.config import ModelTrainerConfig

from sklearn.metrics import accuracy_score,f1_score

class Model_trainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):

        report = {}

        for name, model in models.items():
            logger.info(f"Training {name}")

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            report[name] = {
                "accuracy":acc,
                "f1_score": f1
            }

            logger.info(f"{name} -> Accuracy:{acc}, f1:{f1}")
        
        return report

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):

        try:

            logger.info("Model training started")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC()
            }

            model_report = self.evaluate_models(X_train,y_train)

            best_model_name = max(
                model_report, key = lambda x: model_report[x]['f1_score']
            )

            best_model = models[best_model_name]
            logger.info(f"best model selected: {best_model_name}")

            save_object(self.config.trained_model_file_path, best_model)

            return best_model_name,model_report
        
        except Exception as e:
            raise CustomException(e,sys)