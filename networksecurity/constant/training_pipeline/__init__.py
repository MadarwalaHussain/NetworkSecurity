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