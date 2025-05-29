import os
import sys
import numpy as np
import pandas as pd

"""
Defining Common Constant variables for Training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME = "NetworkSecurity"
ARTIFACT_DIR = "Artifacts"
FILE_NAME = "phising.csv"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data Ingestion realted constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME = "phishing"
DATA_INGESTION_DATABASE_NAME = "NetworkSecurity"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DATA_DIR = "validated"
DATA_VALIDATION_INVALID_DATA_DIR = "invalid"
DATA_VALIDATION_DRFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRFT_REPORT_FILE_NAME = "drift_report.yaml"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME = "transformed_train"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME = "transformed_object"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME = "transformed_test"

# KNN Imputer parameters to replace nans
DATA_TRANSFORMATION_IMPUTER_PARAMS= {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}