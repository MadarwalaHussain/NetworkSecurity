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