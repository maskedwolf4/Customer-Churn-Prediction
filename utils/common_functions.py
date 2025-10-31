import os
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File Not Found in Given Path")

        with open(file_path,'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Read File Succesfully")
            
            return config

    except Exception as e:
        logger.error(f"Error Occured While Reading YAML File ")
        raise CustomException("Failed to Read YAML File ",e)
    

def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error("Failed Laoding Data",e)
        raise CustomException("Failed Loadin data ",e)


