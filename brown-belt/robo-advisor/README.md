# 🤖 Explainable Robo-Advisor API

An end-to-end machine learning system that provides automated, explainable investment risk profiling for financial institutions and investment platforms.

## 📋 Table of Contents
- [Business Problem & Solution](#-business-problem--solution)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Machine Learning Approach](#-machine-learning-approach)
- [API Endpoints](#-api-endpoints)
- [Installation & Setup](#-installation--setup)
- [Docker Deployment](#-docker-deployment)
- [Usage Examples](#-usage-examples)
- [Finance Industry Applications](#-finance-industry-applications)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

## 🎯 Business Problem & Solution

### **Problem Statement**
Financial advisors and robo-advisory platforms need to assess client risk profiles to recommend appropriate investment strategies. Traditional methods are:
- **Manual & Time-consuming**: Advisors spend hours analyzing client profiles
- **Inconsistent**: Different advisors may reach different conclusions for similar clients
- **Non-transparent**: Clients don't understand why they received specific recommendations
- **Not Scalable**: Can't handle thousands of clients simultaneously

### **My Solution**
An **Explainable AI system** that:
- ✅ **Automates** risk profiling in milliseconds
- ✅ **Standardizes** decision-making across all clients
- ✅ **Explains** recommendations with clear reasoning
- ✅ **Scales** to handle unlimited concurrent requests
- ✅ **Integrates** easily with existing financial platforms

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────┐    ┌─────────────────┐
│   Client Data   │───▶│  FastAPI Server  │───▶│ ML Model + SHAP │
│ (Age, Income,   │     │                  │    │   Explainer     │
│  Risk Tolerance,│     │  /predict        │    │                 │
│  Timeline)      │     │  /explain        │    │                 │
└─────────────────┘     └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Risk Profile +  │
                       │  Explanation    │
                       │ (1-5 Scale)     │
                       └─────────────────┘
```

### **Key Components**
1. **Data Layer**: Synthetic client dataset (age, income, risk tolerance, investment horizon)
2. **ML Layer**: Random Forest classifier for risk prediction
3. **Explainability Layer**: SHAP (SHapley Additive exPlanations) for feature importance
4. **API Layer**: FastAPI for REST endpoints
5. **Tracking Layer**: MLflow for experiment tracking
6. **Deployment Layer**: Docker containerization

## 🛠️ Technology Stack

### **Core Technologies**
- **Python 3.10**: Primary programming language
- **FastAPI**: Modern, fast web framework for APIs
- **scikit-learn**: Machine learning library
- **SHAP**: Model explainability framework
- **pandas**: Data manipulation and analysis
- **MLflow**: ML lifecycle management
- **Docker**: Containerization platform

### **Why These Technologies?**
- **FastAPI**: Automatic API documentation, high performance, type validation
- **Random Forest**: Robust, interpretable, handles mixed data types well
- **SHAP**: Industry-standard for ML explainability in finance
- **Docker**: Ensures consistent deployment across environments

## 🧠 Machine Learning Approach

### **Algorithm: Random Forest Classifier**

**Why Random Forest?**
- ✅ **Robust**: Handles outliers and missing data well
- ✅ **Interpretable**: Tree-based models are easier to explain
- ✅ **Feature Importance**: Built-in feature ranking
- ✅ **Non-linear**: Captures complex relationships between features
- ✅ **Ensemble Method**: Reduces overfitting through averaging

### **Model Features**
```python
Input Features:
├── age: Client age (years)
├── income: Annual income (USD)
├── risk_tolerance: Self-assessed risk appetite (1-5 scale)
└── investment_horizon: Investment timeline (years)

Output:
└── risk_level: Recommended investment risk profile (1-5)
    ├── 1: Very Conservative
    ├── 2: Conservative
    ├── 3: Moderate
    ├── 4: Aggressive
    └── 5: Very Aggressive
```

### **Explainability with SHAP**
SHAP (SHapley Additive exPlanations) provides:
- **Feature Importance**: Which factors most influenced the decision
- **Direction of Impact**: Whether each feature increases or decreases risk
- **Magnitude**: How much each feature contributed to the final prediction

**Example Explanation:**
```
"Based on your profile, we recommend a Conservative (Level 2) strategy. 
Factors increasing your risk capacity: your 10-year investment timeline, 
your annual income of $44,440. Factors suggesting lower risk: your age of 30 years."
```

## 🚀 API Endpoints

### **1. Risk Prediction Endpoint**
```http
POST /predict
Content-Type: application/json

{
  "age": 30,
  "income": 75000,
  "risk_tolerance": 3,
  "investment_horizon": 10
}
```

**Response:**
```json
{
  "predicted_risk_level": 3,
  "risk_category": "Moderate"
}
```

### **2. Explainable Prediction Endpoint**
```http
POST /explain
Content-Type: application/json

{
  "age": 30,
  "income": 75000,
  "risk_tolerance": 3,
  "investment_horizon": 10
}
```

**Response:**
```json
{
  "predicted_risk_level": 3,
  "risk_category": "Moderate",
  "user_friendly_explanation": "Based on your profile, we recommend a Moderate (Level 3) investment strategy. Factors increasing your risk capacity: your 10-year investment timeline, your annual income of $75,000. Factors suggesting lower risk: your age of 30 years.",
  "feature_importance": {
    "age": -0.1,
    "income": 0.05,
    "risk_tolerance": 0.03,
    "investment_horizon": 0.08
  },
  "detailed_explanation": {
    "age": {
      "value": 30,
      "shap_value": -0.1,
      "impact": "decreases"
    }
    // ... other features
  }
}
```

### **3. API Documentation**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 💻 Installation & Setup

### **Prerequisites**
- Python 3.10+
- Git
- Docker (optional, for containerized deployment)

### **Local Development Setup**

1. **Clone the Repository**
```bash
git clone <repository-url>
cd robo-advisor
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the Model**
```bash
python train-script.py
```

5. **Run the API**
```bash
python app.py
```

6. **Access the API**
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`

## 🐳 Docker Deployment

### **What is Docker?**
Docker is a containerization platform that packages your application and all its dependencies into a lightweight, portable container. Think of it as a "shipping container" for software.

### **Why Docker for Finance Applications?**

#### **1. Consistency Across Environments**
```bash
# Same behavior everywhere
Developer Machine ✅ → Testing Server ✅ → Production Server ✅
```

#### **2. Rapid Deployment**
```bash
# Traditional deployment
1. Install Python ⏱️
2. Install dependencies ⏱️
3. Configure environment ⏱️
4. Deploy application ⏱️
5. Debug environment issues ⏱️⏱️⏱️

# Docker deployment
1. docker run -p 8000:8000 robo-advisor-api ✅
```

#### **3. Environment Isolation**
- No conflicts with other applications
- Controlled resource usage
- Security isolation

#### **4. Scalability**
```bash
# Run multiple instances easily
docker run -p 8001:8000 robo-advisor-api  # Instance 1
docker run -p 8002:8000 robo-advisor-api  # Instance 2
docker run -p 8003:8000 robo-advisor-api  # Instance 3
```

### **Docker Commands**

#### **Build the Container**
```bash
docker build -t robo-advisor-api .
```

#### **Run the Container**
```bash
docker run -p 8000:8000 robo-advisor-api
```

#### **Container Management**
```bash
# List running containers
docker ps

# Stop a container
docker stop <container_id>

# View logs
docker logs <container_id>

# Execute commands in container
docker exec -it <container_id> /bin/bash
```

### **Production Deployment Benefits**

#### **Cloud Platform Compatibility**
```bash
# Works identically on:
AWS ECS/EKS ✅
Google Cloud Run ✅
Azure Container Instances ✅
Kubernetes ✅
Any Docker-compatible platform ✅
```

#### **CI/CD Integration**
```yaml
# Example GitHub Actions workflow
- name: Build Docker Image
  run: docker build -t robo-advisor-api .
  
- name: Deploy to Production
  run: docker run -d -p 8000:8000 robo-advisor-api
```

## 📚 Usage Examples

### **Using curl**
```bash
# Predict risk level
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 85000,
    "risk_tolerance": 4,
    "investment_horizon": 15
  }'

# Get explanation
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 85000,
    "risk_tolerance": 4,
    "investment_horizon": 15
  }'
```

### **Using Python requests**
```python
import requests

# Client data
client_data = {
    "age": 28,
    "income": 65000,
    "risk_tolerance": 3,
    "investment_horizon": 8
}

# Get prediction
response = requests.post("http://localhost:8000/predict", json=client_data)
print(f"Risk Level: {response.json()['risk_category']}")

# Get explanation
response = requests.post("http://localhost:8000/explain", json=client_data)
print(f"Explanation: {response.json()['user_friendly_explanation']}")
```

### **Integration Example**
```python
# Example: Integrate with a financial planning app
def get_investment_recommendation(client_profile):
    """Get AI-powered investment recommendation"""
    
    # Call the robo-advisor API
    response = requests.post(
        "http://localhost:8000/explain", 
        json=client_profile
    )
    
    recommendation = response.json()
    
    # Use in your application
    return {
        "risk_level": recommendation["risk_category"],
        "explanation": recommendation["user_friendly_explanation"],
        "confidence_factors": recommendation["feature_importance"]
    }
```

## 🏦 Finance Industry Applications

### **1. Robo-Advisory Platforms**
- **Use Case**: Automated portfolio allocation
- **Value**: Scale to millions of clients simultaneously
- **Integration**: Embed as microservice in existing platforms

### **2. Wealth Management Firms**
- **Use Case**: Standardize advisor recommendations
- **Value**: Consistent risk assessment across all advisors
- **Compliance**: Explainable decisions for regulatory requirements

### **3. Bank Onboarding**
- **Use Case**: Instant investment account setup
- **Value**: Reduce customer acquisition time from hours to minutes
- **Experience**: Transparent recommendations build customer trust

### **4. Insurance Companies**
- **Use Case**: Investment-linked insurance product recommendations
- **Value**: Match products to customer risk profiles automatically
- **Scale**: Handle high-volume online applications

### **5. Financial Planning Software**
- **Use Case**: Integrate ML-powered recommendations
- **Value**: Add AI capabilities without building from scratch
- **API-First**: Easy integration with existing systems

### **Regulatory Compliance Benefits**
- ✅ **Explainable**: SHAP explanations satisfy "right to explanation" requirements
- ✅ **Auditable**: All decisions logged with reasoning
- ✅ **Consistent**: Eliminates human bias in recommendations
- ✅ **Documented**: API responses provide audit trail

## 📁 Project Structure

```
robo-advisor/
├── app.py                              # FastAPI application
├── train-script.py                     # Model training script
├── model.joblib                        # Trained ML model
├── synthetic_robo_advisor_data.csv     # Training dataset
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── README.md                           # Project documentation
├── vision.txt                          # Project framework overview
└── mlruns/                            # MLflow experiment tracking
    └── 0/
        ├── meta.yaml
        └── <experiment_files>/
```

### **Key Files Explained**
- **`app.py`**: Main API server with `/predict` and `/explain` endpoints
- **`train-script.py`**: Trains the Random Forest model and logs with MLflow
- **`model.joblib`**: Serialized trained model for API usage
- **`Dockerfile`**: Container configuration for deployment
- **`requirements.txt`**: All Python package dependencies

## 🔧 Development & Testing

### **Model Training**
```bash
# Retrain the model with new data
python train-script.py

# Check MLflow tracking
mlflow ui  # Opens at http://localhost:5000
```

### **API Testing**
```bash
# Start the development server
python app.py

# Run automated tests (if implemented)
pytest tests/

# Test with different client profiles
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age": 25, "income": 45000, "risk_tolerance": 2, "investment_horizon": 5}'
```

### **Docker Development**
```bash
# Build development image
docker build -t robo-advisor-dev .

# Run with volume mounting for live development
docker run -p 8000:8000 -v $(pwd):/app robo-advisor-dev

# Debug inside container
docker exec -it <container_id> /bin/bash
```

## 🚀 Future Enhancements

### **Model Improvements**
- [ ] Feature engineering (income ratios, age brackets)
- [ ] Ensemble methods (XGBoost, Neural Networks)
- [ ] Real-time model retraining
- [ ] A/B testing framework

### **API Enhancements**
- [ ] Authentication and rate limiting
- [ ] Batch prediction endpoints
- [ ] Model versioning support
- [ ] Performance monitoring

### **Business Features**
- [ ] Portfolio allocation recommendations
- [ ] Market condition adjustments
- [ ] ESG (Environmental, Social, Governance) factors
- [ ] Multi-language explanations

### **Deployment & Operations**
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline setup
- [ ] Monitoring and alerting
- [ ] Database integration for client history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure Docker builds successfully

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the MLflow experiments for model insights

---

**Built with ❤️ for the Finance Industry** | **Powered by Explainable AI** | **Docker Ready** 🐳
