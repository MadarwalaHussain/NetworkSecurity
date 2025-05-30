import sys
import os

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_model
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

# importing the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        """
        Initializes the ModelTrainer with the provided configuration.
        
        :param ModelTrainerConfig: Configuration for model training.
        :param data_transformation_artifact: Artifact containing transformed data for training.
        :raises NetworkSecurityException: If an error occurs during initialization.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("Initializing ModelTrainer...")
            logging.info(f"ModelTrainer initialized with config: {self.model_trainer_config}")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def track_mlflow(self, model, classification_metric):
        """
        Tracks the model and its metrics using MLflow.
        
        :param model: The trained model to log.
        :param classification_metric: Metrics for classification to log.
        """
        try:
            logging.info("Logging model and metrics to MLflow...")
            mlflow.set_experiment(experiment_name="network_security_experiment")
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.sklearn.log_model(model, "model")
                logging.info("Model and metrics logged successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Starting model training...")
            # Initialize the model
            models = {
                "Logistic Regression": LogisticRegression(verbose=1),
                "Random Forest Classifier": RandomForestClassifier(verbose=1),
                "Gradient Boosting Classifier": GradientBoostingClassifier(verbose=1),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "KNeighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier()
            }
            params = {
                "Logistic Regression": {},
                "Random Forest Classifier": {"n_estimators": [10,20,50,75,100,150,300]},
                "Gradient Boosting Classifier": {"n_estimators": [10,20,50,75,100,300], "learning_rate":[0.1, 0.01, 0.001, 0.05, 0.5], "subsample":[0.5, 0.75, 1.0]},
                "AdaBoost Classifier": {"n_estimators": [10,20,50,75,100,150,200,250,300], "learning_rate":[0.1, 0.01, 0.001, 0.05, 0.5]},
                "KNeighbors Classifier": {"n_neighbors": [3,5, 7]},
                "Decision Tree Classifier": {"criterion":['gini', 'entropy']}
            }

            model_report:dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)
            logging.info(f"Model evaluation report: {model_report}")

            # Select the best model based on the report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with score: {best_model_score}")
            # Train the best model
            y_train_pred = best_model.predict(x_train)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            logging.info(f"Training completed for model: {best_model_name}")

            # Track the expirements with ML Flow
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Testing completed for model: {best_model_name}")
            self.track_mlflow(best_model, classification_test_metric)
            # Create a NetworkModel instance
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            # Save the trained model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)
            logging.info(f"Trained model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e



    def initiate_model_trainer(self):
        try:
            logging.info("Starting model training process...")
            # Load transformed training and testing data
            train_array = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Transformed training and testing data loaded successfully.")

            # Split the data into features and labels
            x_train, y_train, x_test, y_test = (train_array[:, :-1], train_array[:, -1],test_array[:, :-1], test_array[:, -1])
            logging.info("Data split into features and labels.")
            # Load the preprocessor object
            model = self.train_model(x_train, y_train, x_test, y_test)
            return model
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        