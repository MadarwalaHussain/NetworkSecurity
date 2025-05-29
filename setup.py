'''
The setup.py file is an essential part of Python projects, especially for those that are distributed as packages. 
It contains metadata about the package, such as its name, version, author, and dependencies.
This file is used by tools like setuptools to build and distribute the package.
'''

from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    Reads a requirements file and returns a list of dependencies.
    
    Args:
        file_path (str): The path to the requirements file.
        
    Returns:
        List[str]: A list of dependencies.
    """
    try:
        with open('requirements.txt', 'r') as file:
            requirements = file.readlines()
        # Strip whitespace and newlines from each line
        requirements = [line.strip() for line in requirements if line.strip()]
        # Ignore empty lines and -e.
        requirements = [req for req in requirements if not req.startswith('-e .')]
    except FileNotFoundError:
        print("requirements.txt file not found. Returning an empty list.")

    return requirements

setup(
    name="NetworkSecurity", # Package name
    version="0.0.1", # Package version
    author="Hussain Madarwala", # Author name
    packages=find_packages(), # Automatically find packages in the current directory
    install_requires=get_requirements(), # List of dependencies
    description="A Python package for network security tools and utilities.",

)
