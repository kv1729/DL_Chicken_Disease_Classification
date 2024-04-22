from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
# from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline




def run_data_ingestion_pipeline():
    STAGE_NAME = "Data Ingestion"
    try:
        logger.info(f"----->>> Stage {STAGE_NAME} started <<<-----")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f"----->>> Stage {STAGE_NAME} completed <<<-----\n\nx==========x")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME} stage: {e}")
        raise e


def run_prepare_base_model_pipeline():
    STAGE_NAME = "Prepare base model"
    try:
        logger.info(f"********************") 
        logger.info(f"----->>> Stage {STAGE_NAME} started <<<-----")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f"------>>> Stage {STAGE_NAME} completed <<<------\n\nx=========x")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME} stage: {e}")
        raise e


if __name__ == "__main__":
    run_data_ingestion_pipeline()
    run_prepare_base_model_pipeline()




STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e






# STAGE_NAME = "Evaluation stage"
# try:
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    model_evalution = EvaluationPipeline()
#    model_evalution.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

# except Exception as e:
#         logger.exception(e)
#         raise e