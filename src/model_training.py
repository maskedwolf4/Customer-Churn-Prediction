from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import os
import pickle
from config.path_config import *
from sklearn.metrics import accuracy_score

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self , feature_store:RedisFeatureStore , model_save_path = MODEL_PATH ):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(self.model_save_path , exist_ok=True)
        logger.info("Model Training initialized...")

    def load_data_from_redis(self , entity_ids):
        try:
            logger.info("Extracting data from Redis")

            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning("Feature not found")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis {e}")
            raise CustomException(str(e))
        
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()

            train_entity_ids , test_entity_ids = train_test_split(entity_ids , test_size=0.2 , random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop('Attrition_Flag',axis=1)
            logger.info(X_train.columns)
            X_test = test_df.drop('Attrition_Flag',axis=1)
            y_train = train_df['Attrition_Flag']
            y_test = test_df['Attrition_Flag']

            logger.info("Preparation for Model Training completed")
            return X_train , X_test , y_train, y_test
        
        except Exception as e:
            logger.error(f"Error while preparing data {e}")
            raise CustomException(str(e))
        
    def hyperparamter_tuning(self,X_train,y_train):
        try:
            param_grid = {
                'num_leaves': [5, 20, 31],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 150]
            }
            
            lgb_classifier = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt')
            grid_search = GridSearchCV(estimator=lgb.LGBMClassifier, param_grid=param_grid,
                           scoring='accuracy', cv=5)

            # Fit the model to the training data to search for the best hyperparameters
            grid_search.fit(X_train, y_train)

            # Get the best hyperparameters and their values
            best_params = grid_search.best_params_
            best_hyperparameters = list(best_params.keys())
            best_values = list(best_params.values())

            # Train a LightGBM model with the best hyperparameters
            best_model = lgb_classifier(**best_params)
            best_model.fit(X_train, y_train)


            logger.info(f"Best paramters : {best_params}")
            logger.info(f"Best hyperparamters : {best_hyperparameters}")
            logger.info(f"Best values : {best_values}")
            return grid_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Error while hyperparamter tuning {e}")
            raise CustomException(str(e))
        
    def train_and_evaluate(self , X_train , y_train , X_test , y_test):
        try:
            best_model = self.hyperparamter_tuning(X_train,y_train)

            y_pred = best_model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            self.save_model(best_model)

            return accuracy
        
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomException(str(e))
    
    def save_model(self , model):
        try:
            model_filename = f"{self.model_save_path}lgb_model.pkl"

            with open(model_filename,'wb') as model_file:
                pickle.dump(model , model_file)

            logger.info(f"Model saved at {model_filename}")
        except Exception as e:
            logger.error(f"Error while model saving {e}")
            raise CustomException(str(e))
        
    def run(self):
        try:
            logger.info("Starting Model Training Pipleine....")
            X_train , X_test , y_train, y_test = self.prepare_data()
            accuracy = self.train_and_evaluate(X_train , y_train, X_test , y_test)
            logger.info(f"Accuracy = {accuracy}")

            logger.info("End of Model Training pipeline...")

        except Exception as e:
            logger.error(f"Error while model training pipeline {e}")
            raise CustomException(str(e))
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()

        

            
        
                

