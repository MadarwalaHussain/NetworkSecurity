import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
import pandas as pd
import dill
import pickle


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