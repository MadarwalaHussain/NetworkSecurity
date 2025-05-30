from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, SAVED_MODEL_FILE_NAME

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkModel:
    def __init__(self, preprocessor, model):
        """
        Initializes the NetworkModel with a given model.

        :param model: The machine learning model to be wrapped.
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
            logging.info(f"NetworkModel initialized with model: {self.model}")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def predict(self, x):
        """
        Predicts the output for the given input data.

        :param X: Input data for prediction.
        :return: Predicted output.
        """
        try:
            logging.info(f"Making predictions with model: {self.model}")
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        