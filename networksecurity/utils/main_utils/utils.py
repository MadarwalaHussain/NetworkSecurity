import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str):
    """
    Reads a YAML file and returns its content as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML data.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def  write_yaml_file(file_path: str, content: object, replace: bool = False):
    """
    Writes a dictionary to a YAML file.
    
    :param file_path: Path to the YAML file.
    :param content: Dictionary containing the data to write.
    :param replace: If True, replaces the file if it exists.
    :raises NetworkSecurityException: If an error occurs during file operations.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

def save_numpy_array_data(file_path, array):
    """
    Saves a NumPy array to a file.
    
    :param file_path: Path to the file where the array will be saved.
    :param array: NumPy array to save.
    :raises NetworkSecurityException: If an error occurs during file operations.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str) -> object:
    """
    Loads an object from a file using dill.
    
    :param file_path: Path to the file from which the object will be loaded.
    :return: Loaded object.
    :raises NetworkSecurityException: If an error occurs during file operations.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_object(file_path: str, obj: object):
    """
    Saves an object to a file using dill.
    
    :param file_path: Path to the file where the object will be saved.
    :param obj: Object to save.
    :raises NetworkSecurityException: If an error occurs during file operations.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.
    
    :param file_path: Path to the file from which the array will be loaded.
    :return: Loaded NumPy array.
    :raises NetworkSecurityException: If an error occurs during file operations.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

def evaluate_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None, y_test: np.ndarray = None,
                    models: dict = None, params: dict = None) -> dict:
    """
    Evaluates multiple machine learning models and returns their performance metrics.
    
    :param x_train: Training features.
    :param y_train: Training labels.
    :param x_test: Testing features (optional).
    :param y_test: Testing labels (optional).
    :param models: Dictionary of models to evaluate.
    :param params: Dictionary of parameters for each model.
    :return: Dictionary containing model names and their evaluation scores.
    """
    # Placeholder for the actual implementation
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = {
                "train_score": train_score,
                "test_score": test_score,
                "params": gs.best_params_
            }
            return report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    