#Generate a synthetic dataset with features: age, income, risk_tolerance, investment_horizon, and target risk_level (1-5).

import numpy as np
import pandas as pd

np.random.seed(42)
size = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, size),
    'income': np.random.randint(30000, 200000, size),
    'risk_tolerance': np.random.randint(1, 6, size),        # 1 (low risk) to 5 (high risk)
    'investment_horizon': np.random.randint(1, 31, size),   # years
})

# Simulate risk level (target)
df['risk_level'] = (0.3*(df['income']<80000) + 
                    0.4*(df['age']<35) + 
                    0.6*(df['risk_tolerance']) + 
                    0.1*(df['investment_horizon']>10)).round().astype(int)
df['risk_level'] = df['risk_level'].clip(1, 5)

#Print first few rows of the dataset
print(df.head())


# Train-test split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib

X = df.drop('risk_level', axis=1)
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RoboAdvisorRandomForest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.sklearn.log_model(model, "robo_model")
    joblib.dump(model, "model.joblib")

    print(f"Test Accuracy: {accuracy}")
    print("Model training complete.")
    mlflow.end_run()

# Save the dataset to a CSV file
df.to_csv("synthetic_robo_advisor_data.csv", index=False)

