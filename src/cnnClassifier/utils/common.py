import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns a ConfigBox object.
    
    Args: 
        path_to_yaml (Path): Path to the YAML file.
        
    Raises:
        ValueError: If YAML file is empty.
        e: For other exceptions.
        
    Returns:
        ConfigBox: A ConfigBox object representing the YAML content.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories.
    
    Args: 
        path_to_directories (list): List of paths of directories to be created.
        verbose (bool, optional): Whether to log directory creation information. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Save JSON data to a file.
    
    Args:
        path (Path): Path to JSON file.
        data (dict): Data to be saved in JSON format.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON file data.
    
    Args: 
        path (Path): Path to JSON file.
    
    Returns:
        ConfigBox: Data as class attributes instead of dict.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary data to a file.
    
    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary data from a file.
    
    Args: 
        path (Path): Path to binary file.
    
    Returns:
        Any: Loaded binary data.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Get size of a file in kilobytes (KB).
    
    Args: 
        path (Path): Path of the file.
    
    Returns:
        str: Size of the file in kilobytes (KB).
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"

def decodeImage(imgstring, fileName):
    """Decode base64 image string and save it to a file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
    logger.info(f"Decoded image saved at: {fileName}")

def encodeImageIntoBase64(croppedImagePath):
    """Encode image into base64 format."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
