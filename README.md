# Customer Churn Prediction - End-to-End MLOps Pipeline

An enterprise-grade machine learning system that predicts customer churn using advanced MLOps practices, featuring automated data pipelines, model versioning, drift detection, and scalable deployment architecture.

## Problem Statement

Customer churn is a critical business challenge where companies lose valuable customers to competitors. This system addresses the need for:

- **Proactive Customer Retention**: Identify at-risk customers before they churn
- **Data-Driven Decision Making**: Provide actionable insights with probability scores and risk levels
- **Scalable ML Operations**: Maintain model performance with automated retraining and monitoring
- **Real-time Predictions**: Enable immediate response to customer risk assessment

The solution predicts whether a customer will churn based on their transaction patterns, demographic information, and account behavior, helping businesses implement targeted retention strategies.

## Technical Architecture

![MLOps Architecture for Customer Churn Prediction System](https://github.com/maskedwolf4/Customer-Churn-Prediction/blob/main/assests/mlops_architecture_final.svg)

MLOps Architecture for Customer Churn Prediction System

### Core Components

#### **Data Layer**

- **Raw Data Sources**: Customer transaction and demographic data
- **DVC Integration**: Version-controlled data storage with S3 backend
- **Redis Feature Store**: High-performance feature serving for real-time inference


#### **ML Pipeline Layer**

- **Data Ingestion**: Automated data collection and validation
- **Preprocessing**: Feature engineering, encoding, and scaling
- **Model Training**: Multi-algorithm comparison with hyperparameter optimization
- **Model Registry**: Versioned model storage using DVC


#### **Orchestration Layer**

- **Apache Airflow**: Automated pipeline scheduling and monitoring
- **Training Pipeline**: End-to-end workflow automation


#### **Serving Layer**

- **Flask API**: RESTful endpoints for model inference
- **Model Loading**: Dynamic model retrieval from DVC storage
- **Health Monitoring**: System status and model availability checks


#### **Monitoring Layer**

- **Data Drift Detection**: Alibi Detect KS-Test for distribution changes
- **Logging System**: Comprehensive error tracking and performance metrics
- **Performance Monitoring**: Prediction counts and drift alerting


## Technical Workflow

### **Training Workflow**

1. **Data Ingestion**

```
Raw Data → Validation → Feature Store (Redis)
```

    - Automated data quality checks
    - Feature extraction and storage
    - Data versioning with DVC
2. **Preprocessing Pipeline**

```
Raw Features → Encoding → Scaling → Train/Test Split
```

    - One-hot encoding for categorical variables
    - Standard scaling for numerical features
    - Stratified sampling for balanced datasets
3. **Model Training \& Evaluation**

```
Preprocessed Data → Multiple Algorithms → Model Selection → Registry
```

    - Cross-validation with multiple algorithms
    - Hyperparameter optimization
    - Model performance evaluation
    - Best model registration to DVC
4. **Pipeline Orchestration**

```
Airflow Scheduler → Training DAG → Model Deployment
```

    - Automated retraining schedules
    - Pipeline failure handling
    - Model promotion workflows

### **Inference Workflow**

1. **Real-time Prediction**

```
API Request → Feature Engineering → Model Loading → Prediction → Response
```

    - RESTful API endpoints (`/predict`)
    - Dynamic feature preparation
    - Model loading from DVC storage
    - JSON response with probabilities
2. **Data Drift Detection**

```
New Data → KS-Test → Drift Alert → Model Retraining Trigger
```

    - Kolmogorov-Smirnov statistical test
    - Automatic drift alerting
    - Pipeline trigger for model updates
3. **Monitoring \& Logging**

```
Predictions → Metrics Collection → Dashboard → Alerts
```

    - Performance tracking
    - Error rate monitoring
    - Business metric collection

## Key Features

### **MLOps Best Practices**

- **Reproducible Pipelines**: Version-controlled data, code, and models
- **Automated Testing**: Data validation and model performance checks
- **Continuous Integration**: Automated pipeline execution
- **Model Governance**: Centralized model registry and lifecycle management


### **Production-Ready Architecture**

- **Scalable Deployment**: Docker containerization for cloud deployment
- **High Availability**: Redis feature store for fast access
- **Monitoring**: Comprehensive logging and drift detection
- **API-First Design**: RESTful endpoints for easy integration


### **Advanced ML Capabilities**

- **Feature Engineering**: Automated encoding and scaling
- **Model Selection**: Multi-algorithm comparison framework
- **Drift Detection**: Statistical monitoring for data quality
- **Real-time Inference**: Sub-second prediction response times


## API Endpoints

### **Prediction Endpoint**

```http
POST /predict
Content-Type: application/json

{
    "customer_age": 45,
    "dependent_count": 3,
    "months_on_book": 39,
    "total_relationship_count": 5,
    "months_inactive_12_mon": 1,
    "contacts_count_12_mon": 3,
    "credit_limit": 12691.0,
    "total_revolving_bal": 777,
    "avg_open_to_buy": 11914.0,
    "total_amt_chng_q4_q1": 1.335,
    "total_trans_amt": 1144,
    "total_trans_ct": 42,
    "total_ct_chng_q4_q1": 1.625,
    "avg_utilization_ratio": 0.061,
    "gender": "F",
    "education_level": "Graduate",
    "marital_status": "Single",
    "income_category": "$60K - $80K",
    "card_category": "Blue"
}
```

**Response:**

```json
{
    "prediction": 0,
    "attrition_probability": 0.23,
    "retention_probability": 0.77,
    "status": "Existing Customer",
    "risk_level": "Low"
}
```


### **Health Check**

```http
GET /health
```

**Response:**

```json
{
    "status": "healthy",
    "model_loaded": true,
    "features_count": 32
}
```


## Technology Stack

| Component | Technology | Purpose |
| :-- | :-- | :-- |
| **ML Framework** | scikit-learn | Model training and inference |
| **Data Versioning** | DVC | Data and model version control |
| **Feature Store** | Redis | High-performance feature serving |
| **Orchestration** | Astronomer Airflow | Pipeline automation |
| **Web Framework** | Flask | REST API development |
| **Drift Detection** | Alibi Detect | Statistical monitoring |
| **Containerization** | Docker | Deployment packaging |
| **Storage** | AWS S3 | Model and data storage |
| **Package Management** | UV | Fast Python dependency resolution |

## Quick Start

### **Prerequisites**

- Python 3.12+
- Docker (optional)
- Redis server
- AWS credentials (for DVC)


### **Installation**

```bash
# Clone repository
git clone https://github.com/maskedwolf4/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
uv sync

# Set up DVC remote
dvc remote add -d storage s3://your-bucket/path

# Run training pipeline
python pipeline/training_pipeline.py

# Start the API server
python main.py
```


### **Docker Deployment**

```bash
# Build container
docker build -t churn-prediction .

# Run application
docker run -p 5000:5000 churn-prediction
```


## Development Workflow

1. **Feature Development**: Add new features in `src/` modules
2. **Pipeline Updates**: Modify training workflow in `pipeline/`
3. **Testing**: Run validation scripts and unit tests
4. **Model Training**: Execute pipeline with `python pipeline/training_pipeline.py`
5. **Deployment**: Build Docker container and deploy to cloud

## Monitoring and Maintenance

### **Performance Metrics**

- Prediction latency and throughput
- Model accuracy and drift detection
- API health and error rates


### **Automated Alerts**

- Data drift detection triggers
- Model performance degradation
- System health monitoring


### **Maintenance Tasks**

- Regular model retraining schedules
- Feature store optimization
- Infrastructure scaling adjustments

This MLOps pipeline provides a robust foundation for production machine learning systems, combining best practices in data engineering, model development, and operational excellence to deliver reliable customer churn predictions at scale.

## **Frontend Part of Application**

![Frontend](https://github.com/maskedwolf4/Customer-Churn-Prediction/blob/main/assests/image.png)



