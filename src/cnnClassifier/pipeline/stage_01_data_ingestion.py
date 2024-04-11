from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f"----->>> Stage {STAGE_NAME} started <<<-----")
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info(f"----->>> Stage {STAGE_NAME} completed <<<-----\n\nx=========x")
        except Exception as e:
            logger.exception(f"Error occurred in stage {STAGE_NAME}: {e}")
            raise

if __name__ == '__main__':
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
