import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import numpy as np

# Load the data
df = pd.read_csv('german_credit_data.csv')

# print(df.head())
# print(df.info())

# # Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Encode categorical features
X_encoded = X.copy()
label_encoders = {}

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# # Make predictions
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")