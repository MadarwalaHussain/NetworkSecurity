import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact:DataTransformationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    @staticmethod
    def read_data(file_path):
        """
        Reads a CSV file and returns a DataFrame.
        
        :param file_path: Path to the CSV file.
        :return: DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    

    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates a data transformation pipeline with KNNImputer.
        
        :return: A Pipeline object for data transformation.
        """
        try:
            logging.info("Creating data transformation pipeline with KNNImputer.")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline = Pipeline(steps=[('imputer', imputer)])
            logging.info("Data transformation pipeline created successfully.")
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"{'>>' * 30} Data Transformation {'<<' * 30}")
            # Read the validated data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Training DataFrame
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)  # Replace -1 with 0 in target column
            logging.info("Target feature in training data transformed successfully.")
            # Testing DataFrame
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)  # Replace -1 with 0 in target column
            logging.info("Target feature in testing data transformed successfully.")

            # Create the data transformation pipeline
            data_transformation_pipeline = self.get_data_transformer_object()
            logging.info("Fitting the data transformation pipeline on training data.")
            data_transformation_pipeline.fit(input_feature_train_df)
            logging.info("Data transformation pipeline fitted successfully.")
            # Transform the training and testing data
            logging.info("Transforming the training data.")
            transformed_input_feature_train = data_transformation_pipeline.transform(input_feature_train_df)
            logging.info("Training data transformed successfully.")

            logging.info("Transforming the testing data.")
            transformed_input_feature_test = data_transformation_pipeline.transform(input_feature_test_df)
            logging.info("Testing data transformed successfully.")

            # Save the transformed data as numpy arrays
            train_arr = np.c_[transformed_input_feature_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test, np.array(target_feature_test_df)]

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            logging.info("Transformed training data saved successfully.")
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            logging.info("Transformed testing data saved successfully.")
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=data_transformation_pipeline
            )

            # Create and return the DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation artifact created successfully.")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    
