from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            logging.info(f"{'>>' * 30} Data Validation {'<<' * 30}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
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
    

    def validate_data_schema(self, dataframe):
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Expected number of columns: {number_of_columns}")
            if len(dataframe.columns) != number_of_columns:
                raise ValueError(f"Dataframe has {len(dataframe.columns)} columns, expected {number_of_columns} columns.")
                return False
            else:
                logging.info("Data schema validation passed: Number of columns match expected schema.")
                return True
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def detect_data_drift(self, base_data, current_data, threshold=0.05):
        try:
            status=True
            report = {}
            for column in base_data.columns:
                d1 = base_data[column]
                d2 = current_data[column]
                ks_2samp_result = ks_2samp(d1, d2)
                logging.info(f"KS Test for column '{column}': statistic={ks_2samp_result.statistic}, pvalue={ks_2samp_result.pvalue}")
                if threshold <=ks_2samp_result.pvalue:
                    is_drifted = False
                else:
                    is_drifted = True
                    status=False
                report.update({column: {
                    "drift_status": is_drifted,
                    "p_value": ks_2samp_result.pvalue,
                    "statistic": ks_2samp_result.statistic
                }})
            logging.info(f"Data Drift Report: {report}")
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            logging.info("Data drift report generated successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self):
        try:
            logging.info("Starting data validation process")
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the training and testing data
            train_data = DataValidation.read_data(train_file_path)
            test_data = DataValidation.read_data(test_file_path)
            logging.info("Data validation initiated successfully")
            # Validate data schema
            is_train_valid = self.validate_data_schema(train_data)
            if not is_train_valid:
                raise ValueError("Training data schema validation failed.")
            is_test_valid = self.validate_data_schema(test_data)
            if not is_test_valid:
                raise ValueError("Testing data schema validation failed.")
            logging.info("Data schema validation passed for both training and testing data")    

            # Validating Data Drift
            status = self.detect_data_drift(train_data, test_data)
            logging.info("Data drift detection completed successfully")
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_data.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_data.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            logging.info("Validated data files saved successfully")

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e