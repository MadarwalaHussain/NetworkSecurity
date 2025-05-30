from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        # Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        # Create a DataIngestion instance with the configuration
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        logging.info("Starting data ingestion process...")
        # Export the collection to a DataFrame
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data ingestion completed successfully.")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Starting data validation process...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully.")
        print(data_validation_artifact)

        logging.info("Starting data transformation process...")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully.")
        print(data_transformation_artifact)

        logging.info("Model training process is about to start...")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        mode_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config
        )
        model_trainer_artifact = mode_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully.")
        print(model_trainer_artifact)



    except Exception as e:
        raise NetworkSecurityException(
            error_message=e,
            error_details=sys
        ) from e
