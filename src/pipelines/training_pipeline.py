import os 
import sys 
from src.logger import logging
from src.exception import CustomException
 
import pandas as pd

 
from src.components import data_ingestion
from src.components import data_transformation
from src.components import model_trainer


if __name__ == '__main__':
    obj = data_ingestion.DataIngestion()
    train_data_path, test_data_path =  obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transformation = data_transformation.DataTransformation()

    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = model_trainer.ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)