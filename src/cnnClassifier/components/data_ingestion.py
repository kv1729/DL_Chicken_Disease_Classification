import os
import urllib.request as request
import logging
import zipfile
from pathlib import Path
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger.info(f"File downloaded successfully. File size: {get_size(Path(self.config.local_data_file))}")
            else:
                logger.info(f"File already exists. File size: {get_size(Path(self.config.local_data_file))}")
        except Exception as e:
            logger.error(f"Error occurred while downloading file: {e}")

    def extract_zip_file(self):
        """
        Extracts the contents of the ZIP file into the data directory
        """
        unzip_path = self.config.unzip_dir
        try:
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info("ZIP file extracted successfully.")
        except Exception as e:
            logger.error(f"Error occurred while extracting ZIP file: {e}")
