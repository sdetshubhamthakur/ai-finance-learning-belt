# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# print("Path to dataset files:", path)

import pandas as pd

df = pd.read_csv("creditcard.csv")
print("First 5 rows: \n", df.head())
print(df['Class'].value_counts())

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

#Fraud is rare! Use oversampling (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Resampled training set size:", X_train_resampled.shape)
print(pd.Series(y_train_resampled).value_counts())      #This should balance the classes

    
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, recall_score

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred, pos_label=1))  # Focus on recall for class 1 (fraud)
