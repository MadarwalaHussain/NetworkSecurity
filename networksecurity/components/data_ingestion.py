from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Importing Data Ingestion configuration
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.client[self.data_ingestion_config.database_name]
            self.collection = self.database[self.data_ingestion_config.collection_name]
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )
        
    def export_collection_to_dataframe(self):

        """
        Exporting MongoDB collection to a Pandas DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the collection data.
        """
        logging.info("Exporting collection to DataFrame.")
        try:
            logging.info("Exporting collection to DataFrame.")
            # Fetching data from MongoDB collection
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            logging.info(f"Fetching data from collection: {collection_name} in database: {database_name}.")
            dataframe = pd.DataFrame(list(collection.find()))
            if '_id' in dataframe.columns:
                dataframe.drop('_id', axis=1, inplace=True)
            dataframe = dataframe.replace({"na":np.nan})
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )

    def export_data_to_feature_store(self, dataframe: pd.DataFrame):
        """
        Exporting DataFrame to feature store.
        Args:
            dataframe (pd.DataFrame): DataFrame to be exported.
        """
        logging.info("Exporting DataFrame to feature store.")
        try:
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            # Logging the successful export
            logging.info(f"Data exported to feature store at {self.data_ingestion_config.feature_store_file_path}.")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splitting DataFrame into training and testing datasets.
        Args:
            dataframe (pd.DataFrame): DataFrame to be split.
        Returns:
            tuple: Training and testing DataFrames.
        """
        logging.info("Splitting data into train and test sets.")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}.")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            # Creating directories if they do not exist
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test sets to CSV files.")

            # Saving the train and test sets to CSV files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Train set saved at {self.data_ingestion_config.training_file_path}.")
            logging.info(f"Test set saved at {self.data_ingestion_config.testing_file_path}.")
            return train_set, test_set
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started.")
            # Fetching data from MongoDB collection
            dataframe = self.export_collection_to_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            logging.info("Data ingestion completed successfully.")

            train_set, test_set = self.split_data_as_train_test(dataframe)
            logging.info("Data ingestion process completed successfully.")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )