from sklearn.impute import SimpleImputer #handling missing values 
from sklearn.preprocessing import StandardScaler #handling features scaling 
from sklearn.preprocessing import OrdinalEncoder #ordinal encoding


##for pipelines 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #to combine 2 different pipelines 
import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd


from src.exception1 import CustomException
from src.logger import logging

from src.utils import save_object

# data transformation config 
@dataclass 
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')



# data ingestionconfig class 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation()

    def get_data_transformation_object(self):
        try:
            logging.info("data transformation initiated")

            #defining which column should be ordinal encoded and which should be scaled
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']


            # Defining the cusotm ranking for each of the ordinal variables 
            cut_categories = ['Fair','Good','Very Good','Premium',  'Ideal'  ]
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J' ]
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2',  'VS1',   'VVS2', 'VVS1', 'IF']


            logging.info("Pipeline Initiated")


            # Numerical Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  #handling the missing values with median 
                    ('scaler', StandardScaler())
                ]
            )


            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar', StandardScaler())

                ]
            )


            #this combines the numerical pipeline and categorical pipeline and numerical pipeline
            preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols)

            ])

            return preprocessor
        
            logging.info("Pipeline completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
    



    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            #Reading train and test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train dataframe head:\n{train_df.head().to_string()}')
            logging.info(f'Test dataframe head:\n{test_df.head().to_string()}')
            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()


            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            #dividing features into independent and dependent feature

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

        
            # applying the transformation 
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets")



            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor pickel is created and saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)

