import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_data_path , test_data_path , feature_store : RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data=None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train=None
        self.y_test = None

        self.X_scaled = None

        self.feature_store = feature_store
        logger.info("Your Data Processing is intialized...")
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Read the data sucesfully")
        except Exception as e:
            logger.error(f"Error while reading data {e}")
            raise CustomException(str(e))
    

    def preprocess_data(self):
        try:
            
            self.data['Attrition_Flag'] = self.data['Attrition_Flag'].map({
                'Existing Customer': 0,
                'Attrited Customer': 1
            })

            # Gender: binary label encoding
            self.data['Gender'] = self.data['Gender'].map({'M': 1, 'F': 0})

            # Remaining categoric variables: OneHotEncoding
            self.data = pd.get_dummies(self.data, columns=[
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
            ], drop_first=True)

            self.data = self.data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
            'CLIENTNUM'],axis='columns', inplace=True)

            logger.info("Data Preprocessing done...")

        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            raise CustomException(str(e))
    
    def scale_data(self):
        try:
            X = self.data.drop(["Attrited Customer"])
            y = self.data['Attrited Customer']

            scaler = StandardScaler()
            
            self.X_scaled= scaler.fit_transform(X=X)

            logger.info("Scaled data sucesfully...")

        except Exception as e:
            logger.error(f"Error while scaling data {e}")
            raise CustomException(str(e))
    
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for idx,row in self.data.iterrows():
                entity_id = row["PassengerId"]
                features = {
                    "Attrition_Flag" : row['Attrition_Flag'],
                    "Contacts_Count_12_mon" : row["Contacts_Count_12_mon"],
                    "Months_Inactive_12_mon" : row["Months_Inactive_12_mon"],
                    "Education_Level_Doctorate" : row["Education_Level_Doctorate"],
                    "Income_Category_Less than $40K" : row["Income_Category_Less than $40K"],
                    "Marital_Status_Single": row["Marital_Status_Single"],
                    "Dependent_count" : row["Dependent_count"],
                    "Customer_Age" : row["Customer_Age"],
                    "Months_on_book" : row["Months_on_book"],
                    "Education_Level_Post-Graduate" : row["Education_Level_Post-Graduate"],
                    "Card_Category_Platinum" : row["Card_Category_Platinum"],
                    "Education_Level_Unknown" : row["Education_Level_Unknown"],
                    "Marital_Status_Unknown" : row["Marital_Status_Unknown"],
                    "Income_Category_Unknown" : row["Income_Category_Unknown"],
                    "Card_Category_Gold" : row["Card_Category_Gold"],
                    "Avg_Open_To_Buy" : row["Avg_Open_To_Buy"],
                    "Education_Level_Uneducated" : row["Education_Level_Uneducated"],
                    "Income_Category_$80K - $120K" : row["Income_Category_$80K - $120K"],
                    "Card_Category_Silver" : row["Card_Category_Silver"],
                    "Education_Level_Graduate" : row["Education_Level_Graduate"],
                    "Income_Category_$40K - $60K" : row["Income_Category_$40K - $60K"],
                    "Education_Level_High School" : row["Education_Level_High School"],
                    "Marital_Status_Married" : row["Marital_Status_Married"],
                    "Credit_Limit" : row["Credit_Limit"],
                    "Income_Category_$60K - $80K" : row["Income_Category_$60K - $80K"],
                    "Gender" : row["Gender"],
                    "Total_Amt_Chng_Q4_Q1" : row["Total_Amt_Chng_Q4_Q1"],
                    "Total_Relationship_Count" : row["Total_Relationship_Count"],
                    "Total_Trans_Amt" : row["Total_Trans_Amt"],
                    "Avg_Utilization_Ratio" : row["Avg_Utilization_Ratio"],
                    "Total_Revolving_Bal" : row["Total_Revolving_Bal"],
                    "Total_Ct_Chng_Q4_Q1" : row["Total_Ct_Chng_Q4_Q1"],
                    "Total_Trans_Ct" : row["Total_Trans_Ct"]
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been feeded into Feature Store..")
        except Exception as e:
            logger.error(f"Error while feature storing data {e}")
            raise CustomException(str(e))
        
    def retrive_feature_redis_store(self,entity_id):
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None
    
    def run(self):
        try:
            logger.info("Starting our Data Processing Pipleine...")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_feature_in_redis()

            logger.info("End of pipeline Data Processing...")

        except Exception as e:
            logger.error(f"Error while Data Processing Pipleine {e}")
            raise CustomException(str(e))
        
if __name__=="__main__":
    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()

    print(data_processor.retrive_feature_redis_store(entity_id=332))
        


