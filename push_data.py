import os 
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from dotenv import load_dotenv
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where() # Get the path to the CA bundle (Certificate Authority bundle)

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data = data.reset_index(drop=True)
            json_data = list(json.loads(data.T.to_json()).values())
            return json_data
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )
    def insert_data_to_mongo(self, records, database, collection_name):
        try:
            self.records = records
            self.database = database
            self.collection_name = collection_name
            self.client = pymongo.MongoClient(MONGO_DB_URL)

            self.database= self.client[self.database]
            self.collection = self.database[self.collection_name]

            self .collection.insert_many(self.records)
            logging.info(f"Data inserted successfully into {self.collection_name} collection in {self.database} database.")
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(
                error_message=e,
                error_details=sys
            )
        
if __name__ == "__main__":
    try:
        FILE_PATH = 'Network_Data\phising.csv'
        DATABASE = 'NetworkSecurity'
        COLLECTION_NAME = 'phishing'
        network_data_extract = NetworkDataExtract()
        json_data = network_data_extract.csv_to_json_convertor(FILE_PATH)
        logging.info(f"Converted CSV data to JSON format. Number of records: {len(json_data)}")
        no_of_records = network_data_extract.insert_data_to_mongo(
            records=json_data,
            database=DATABASE,
            collection_name=COLLECTION_NAME
        )
        print(f"Number of records inserted: {no_of_records}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise NetworkSecurityException(
            error_message=e,
            error_details=sys
        )