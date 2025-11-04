from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import dvc.api
import os
from src.logger import get_logger
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler
from alibi_detect.cd import KSDrift
from prometheus_client import start_http_server, Counter, Gauge

logger = get_logger(__name__)

app = Flask(__name__)

prediction_count = Counter('prediction_count' , " Number of prediction count" )
drift_count = Counter('drift_count' , "Numer of times data drift is detected")

# Global variable to store the model
model = None

# Define all feature columns in the correct order
FEATURE_COLUMNS = [
    "Contacts_Count_12_mon",
    "Months_Inactive_12_mon",
    "Education_Level_Doctorate",
    "Income_Category_Less than $40K",
    "Marital_Status_Single",
    "Dependent_count",
    "Customer_Age",
    "Months_on_book",
    "Education_Level_Post-Graduate",
    "Card_Category_Platinum",
    "Education_Level_Unknown",
    "Marital_Status_Unknown",
    "Income_Category_Unknown",
    "Card_Category_Gold",
    "Avg_Open_To_Buy",
    "Education_Level_Uneducated",
    "Income_Category_$80K - $120K",
    "Card_Category_Silver",
    "Education_Level_Graduate",
    "Income_Category_$40K - $60K",
    "Education_Level_High School",
    "Marital_Status_Married",
    "Credit_Limit",
    "Income_Category_$60K - $80K",
    "Gender",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Relationship_Count",
    "Total_Trans_Amt",
    "Avg_Utilization_Ratio",
    "Total_Revolving_Bal",
    "Total_Ct_Chng_Q4_Q1",
    "Total_Trans_Ct"
]

feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    all_features_df = pd.DataFrame.from_dict(all_features , orient='index')[FEATURE_COLUMNS]

    scaler.fit(all_features_df)
    return scaler.transform(all_features_df)


historical_data = fit_scaler_on_ref_data()
ksd = KSDrift(x_ref=historical_data , p_val=0.05)

def load_model_from_dvc():
    """Load model from DVC S3 storage"""
    global model
    try:
        # Using dvc.api.read() to load directly from S3
        data = dvc.api.read(
            'model.pkl',
            repo='https://github.com/maskedwolf4/Customer-Churn-Prediction',
            mode='rb'
        )
        model = pickle.loads(data)
        print("✓ Model loaded successfully from DVC storage")
    except Exception as e:
        print(f"Warning: Could not load from DVC: {e}")
        # Fallback to local file
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded from local file")
        else:
            print("✗ Model not found!")

def prepare_features(form_data):
    """Convert form data to model input with one-hot encoding"""
    
    # Initialize all features with 0
    features = {col: 0 for col in FEATURE_COLUMNS}
    
    # Set numeric features directly
    features['Contacts_Count_12_mon'] = int(form_data.get('contacts_count', 0))
    features['Months_Inactive_12_mon'] = int(form_data.get('months_inactive', 0))
    features['Dependent_count'] = int(form_data.get('dependent_count', 0))
    features['Customer_Age'] = int(form_data.get('customer_age', 0))
    features['Months_on_book'] = int(form_data.get('months_on_book', 0))
    features['Avg_Open_To_Buy'] = float(form_data.get('avg_open_to_buy', 0))
    features['Credit_Limit'] = float(form_data.get('credit_limit', 0))
    features['Total_Amt_Chng_Q4_Q1'] = float(form_data.get('total_amt_chng', 0))
    features['Total_Relationship_Count'] = int(form_data.get('total_relationship_count', 0))
    features['Total_Trans_Amt'] = float(form_data.get('total_trans_amt', 0))
    features['Avg_Utilization_Ratio'] = float(form_data.get('avg_utilization_ratio', 0))
    features['Total_Revolving_Bal'] = float(form_data.get('total_revolving_bal', 0))
    features['Total_Ct_Chng_Q4_Q1'] = float(form_data.get('total_ct_chng', 0))
    features['Total_Trans_Ct'] = int(form_data.get('total_trans_ct', 0))
    
    # Gender: M=1, F=0
    features['Gender'] = 1 if form_data.get('gender') == 'M' else 0
    
    # One-hot encode Education Level
    education = form_data.get('education_level')
    if education == 'Doctorate':
        features['Education_Level_Doctorate'] = 1
    elif education == 'Post-Graduate':
        features['Education_Level_Post-Graduate'] = 1
    elif education == 'Graduate':
        features['Education_Level_Graduate'] = 1
    elif education == 'High School':
        features['Education_Level_High School'] = 1
    elif education == 'Uneducated':
        features['Education_Level_Uneducated'] = 1
    elif education == 'Unknown':
        features['Education_Level_Unknown'] = 1
    
    # One-hot encode Marital Status
    marital = form_data.get('marital_status')
    if marital == 'Single':
        features['Marital_Status_Single'] = 1
    elif marital == 'Married':
        features['Marital_Status_Married'] = 1
    elif marital == 'Unknown':
        features['Marital_Status_Unknown'] = 1
    
    # One-hot encode Income Category
    income = form_data.get('income_category')
    if income == 'Less than $40K':
        features['Income_Category_Less than $40K'] = 1
    elif income == '$40K - $60K':
        features['Income_Category_$40K - $60K'] = 1
    elif income == '$60K - $80K':
        features['Income_Category_$60K - $80K'] = 1
    elif income == '$80K - $120K':
        features['Income_Category_$80K - $120K'] = 1
    elif income == 'Unknown':
        features['Income_Category_Unknown'] = 1
    
    # One-hot encode Card Category
    card = form_data.get('card_category')
    if card == 'Platinum':
        features['Card_Category_Platinum'] = 1
    elif card == 'Gold':
        features['Card_Category_Gold'] = 1
    elif card == 'Silver':
        features['Card_Category_Silver'] = 1
    
    return features

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Prepare features with one-hot encoding
        features_dict = prepare_features(data)
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        ##### Data Drift Detection
        features_scaled = scaler.transform(input_df)

        drift = ksd.predict(features_scaled)
        print("Drift Response : ",drift)

        drift_response = drift.get('data',{})
        is_drift = drift_response.get('is_drift' , None)

        if is_drift is not None and is_drift==1:
            print("Drift Detected....")
            logger.info("Drift Detected....")

            drift_count.inc()
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_count.inc()
        probability = model.predict_proba(input_df)[0]
        
        # Format response
        result = {
            'prediction': int(prediction),
            'attrition_probability': float(probability[1]),
            'retention_probability': float(probability[0]),
            'status': 'Attrited Customer' if prediction == 1 else 'Existing Customer',
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(FEATURE_COLUMNS)
    })

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest() , content_type='text/plain')

if __name__ == '__main__':
    start_http_server()
    # Load model before starting the server
    load_model_from_dvc()
    app.run(debug=True, host='0.0.0.0', port=5000)
