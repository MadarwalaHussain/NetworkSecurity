from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionCofig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        # Initialize the training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionCofig(training_pipeline_config=training_pipeline_config)

        # Create a DataIngestion instance with the configuration
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        logging.info("Starting data ingestion process...")
        # Export the collection to a DataFrame
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        # Export the DataFrame to the feature store
        # data_ingestion.export_data_to_feature_store(dataframe=dataframe)

    except Exception as e:
        raise NetworkSecurityException(
            error_message=e,
            error_details=sys
        ) from e