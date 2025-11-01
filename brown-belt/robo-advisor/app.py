# Build explainable API with FastAPI and SHAP
# Two endpoints:

# /predict returns the predicted risk level
# /explain returns SHAP feature importances

from fastapi import FastAPI
import shap
import uvicorn
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Robo-Advisor API with SHAP Explanations")

# Load the trained model
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)

class ClientData(BaseModel):
    age: int
    income: int
    risk_tolerance: int
    investment_horizon: int

@app.post("/predict")
def predict_risk_level(data: ClientData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    
    # Risk level categories
    risk_levels = {
        1: "Very Conservative",
        2: "Conservative", 
        3: "Moderate",
        4: "Aggressive",
        5: "Very Aggressive"
    }
    
    return {
        "predicted_risk_level": int(prediction),
        "risk_category": risk_levels[int(prediction)]
    }

@app.post("/explain")
def explain_risk_level(data: ClientData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        
        # Get prediction for context
        prediction = model.predict(input_df)[0]
        prediction = int(prediction)  # Ensure it's a scalar
        
        # Generate SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For multi-class, get SHAP values for predicted class
            predicted_class_idx = prediction - 1  # Assuming classes 1-5, convert to 0-4
            if predicted_class_idx < 0 or predicted_class_idx >= len(shap_values):
                predicted_class_idx = 0  # Default to first class if index is out of bounds
            shap_vals = shap_values[predicted_class_idx][0]
        else:
            shap_vals = shap_values[0]
        
        # Ensure shap_vals is a 1D array
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals.flatten()
        
        # Create feature importance dictionary
        feature_names = input_df.columns.tolist()
        feature_importance = dict(zip(feature_names, shap_vals.tolist()))
        
        # Generate user-friendly explanation
        risk_levels = {
            1: "Very Conservative",
            2: "Conservative", 
            3: "Moderate",
            4: "Aggressive",
            5: "Very Aggressive"
        }
        
        # Find the most influential features
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f for f, v in sorted_features if v > 0][:2]
        top_negative = [f for f, v in sorted_features if v < 0][:2]
        
        # Create explanation text
        explanation_text = f"Based on your profile, we recommend a {risk_levels[prediction]} (Level {prediction}) investment strategy. "
        
        if top_positive:
            explanation_text += f"Factors increasing your risk capacity: "
            for feature in top_positive:
                value = input_df[feature].iloc[0]
                if feature == "age":
                    explanation_text += f"your age of {value} years, "
                elif feature == "income":
                    explanation_text += f"your annual income of ${value:,}, "
                elif feature == "risk_tolerance":
                    explanation_text += f"your risk tolerance level of {value}/5, "
                elif feature == "investment_horizon":
                    explanation_text += f"your {value}-year investment timeline, "
            explanation_text = explanation_text.rstrip(", ") + ". "
        
        if top_negative:
            explanation_text += f"Factors suggesting lower risk: "
            for feature in top_negative:
                value = input_df[feature].iloc[0]
                if feature == "age":
                    explanation_text += f"your age of {value} years, "
                elif feature == "income":
                    explanation_text += f"your annual income of ${value:,}, "
                elif feature == "risk_tolerance":
                    explanation_text += f"your risk tolerance level of {value}/5, "
                elif feature == "investment_horizon":
                    explanation_text += f"your {value}-year investment timeline, "
            explanation_text = explanation_text.rstrip(", ") + ". "
        
        return {
            "predicted_risk_level": prediction,
            "risk_category": risk_levels[prediction],
            "user_friendly_explanation": explanation_text,
            "feature_importance": feature_importance,
            "detailed_explanation": {
                name: {
                    "value": float(input_df[name].iloc[0]),
                    "shap_value": float(importance),
                    "impact": "increases" if importance > 0 else "decreases"
                }
                for name, importance in feature_importance.items()
            }
        }
    except Exception as e:
        return {"error": f"Explanation generation failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)