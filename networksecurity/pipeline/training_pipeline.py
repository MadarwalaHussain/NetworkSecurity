import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig

from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact


class TrainingPipeline:
    def __init__(self):
        """
        Initializes the training pipeline with the provided configuration.
        
        :param training_pipeline_config: Configuration for the training pipeline.
        :raises NetworkSecurityException: If an error occurs during initialization.
        """
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            logging.info("Training Pipeline initialized with config: {}".format(self.training_pipeline_config))
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process...")
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("Starting data validation process...")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed successfully.")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Starting data transformation process...")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_validation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully.")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Starting model training process...")
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            mode_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=self.model_trainer_config)

            model_trainer_artifact = mode_trainer.initiate_model_trainer()
            logging.info("Model training completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline...")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)
            logging.info("Training pipeline completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e