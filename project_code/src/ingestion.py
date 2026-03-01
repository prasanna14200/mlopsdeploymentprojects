import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.local_data_path = self.config["local_data_path"]   # path to kaggle csv
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info("Data Ingestion initialized successfully")

    def load_and_store_raw(self):
        try:
            logger.info("Loading dataset from local path")

            data = pd.read_csv(self.local_data_path)

            # Save raw copy inside artifacts/raw
            data.to_csv(RAW_FILE_PATH, index=False)

            logger.info(f"Raw data saved to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while loading raw dataset")
            raise CustomException("Failed to load raw dataset", e)

    def split_data(self):
        try:
            logger.info("Starting train-test split")

            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_test_ratio,
                random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error while splitting dataset")
            raise CustomException("Failed to split dataset", e)

    def run(self):
        try:
            logger.info("Starting Data Ingestion Process")

            self.load_and_store_raw()
            self.split_data()

            logger.info("Data Ingestion Completed Successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")

        finally:
            logger.info("Data Ingestion Pipeline Finished")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()