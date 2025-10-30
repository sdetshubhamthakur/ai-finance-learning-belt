import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import time

# Load the data
print("Loading creditcard dataset...")
df = pd.read_csv("creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Apply SMOTE for handling class imbalance
print("\nApplying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training set size: {X_train_resampled.shape}")
print(f"Resampled class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

# Scale features for Logistic Regression (RandomForest doesn't need scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
    # Metrics
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall AUC (better for imbalanced datasets)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'model': model,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cv_score': cv_scores.mean(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

# Initialize models
print("\n" + "="*60)
print("FRAUD DETECTION: RandomForest vs Logistic Regression")
print("="*60)

# RandomForest (doesn't need scaled features)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Logistic Regression (needs scaled features)
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance
)

# Evaluate both models
rf_results = evaluate_model(rf_model, X_train_resampled, X_test, y_train_resampled, y_test, "Random Forest")
lr_results = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train_resampled, y_test, "Logistic Regression")

# Comparison Summary
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

comparison_data = {
    'Metric': ['ROC-AUC', 'Precision-Recall AUC', 'Cross-Val ROC-AUC', 'Training Time (s)', 'Prediction Time (s)'],
    'Random Forest': [
        f"{rf_results['roc_auc']:.4f}",
        f"{rf_results['pr_auc']:.4f}",
        f"{rf_results['cv_score']:.4f}",
        f"{rf_results['training_time']:.4f}",
        f"{rf_results['prediction_time']:.4f}"
    ],
    'Logistic Regression': [
        f"{lr_results['roc_auc']:.4f}",
        f"{lr_results['pr_auc']:.4f}",
        f"{lr_results['cv_score']:.4f}",
        f"{lr_results['training_time']:.4f}",
        f"{lr_results['prediction_time']:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Feature importance for Random Forest
print(f"\nTop 10 Most Important Features (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_results['model'].feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Analysis and Recommendations
print(f"\n" + "="*60)
print("ANALYSIS & RECOMMENDATIONS")
print("="*60)

print(f"""
FRAUD DETECTION CONTEXT:
- Dataset is highly imbalanced ({(df['Class']==0).sum()} normal vs {(df['Class']==1).sum()} fraud cases)
- False Negatives (missing fraud) are more costly than False Positives
- Model interpretability vs performance trade-off is important

PERFORMANCE COMPARISON:
""")

if rf_results['roc_auc'] > lr_results['roc_auc']:
    print(f"✓ Random Forest has higher ROC-AUC ({rf_results['roc_auc']:.4f} vs {lr_results['roc_auc']:.4f})")
else:
    print(f"✓ Logistic Regression has higher ROC-AUC ({lr_results['roc_auc']:.4f} vs {rf_results['roc_auc']:.4f})")

if rf_results['pr_auc'] > lr_results['pr_auc']:
    print(f"✓ Random Forest has higher PR-AUC ({rf_results['pr_auc']:.4f} vs {lr_results['pr_auc']:.4f})")
else:
    print(f"✓ Logistic Regression has higher PR-AUC ({lr_results['pr_auc']:.4f} vs {lr_results['pr_auc']:.4f})")

print(f"""
SPEED COMPARISON:
- Random Forest training: {rf_results['training_time']:.4f}s
- Logistic Regression training: {lr_results['training_time']:.4f}s
- Random Forest prediction: {rf_results['prediction_time']:.4f}s  
- Logistic Regression prediction: {lr_results['prediction_time']:.4f}s

RECOMMENDATIONS FOR FRAUD DETECTION:
""")

# Determine better model based on PR-AUC (more relevant for imbalanced data)
if rf_results['pr_auc'] > lr_results['pr_auc']:
    better_model = "Random Forest"
    worse_model = "Logistic Regression"
    better_score = rf_results['pr_auc']
else:
    better_model = "Logistic Regression" 
    worse_model = "Random Forest"
    better_score = lr_results['pr_auc']

print(f"""
1. PRIMARY RECOMMENDATION: {better_model}
   - Higher Precision-Recall AUC ({better_score:.4f}) - crucial for imbalanced fraud data
   - Better at distinguishing fraud from normal transactions
   
2. WHEN TO USE RANDOM FOREST:
   - When you need feature importance insights
   - When you have non-linear relationships in data
   - When model interpretability is less critical
   - When you can afford longer training times
   
3. WHEN TO USE LOGISTIC REGRESSION:
   - When you need fast predictions in real-time
   - When model interpretability is crucial (coefficient analysis)
   - When you have limited computational resources
   - When you need probabilistic outputs that are well-calibrated

4. FRAUD DETECTION SPECIFIC CONSIDERATIONS:
   - Precision-Recall AUC is more important than ROC-AUC for imbalanced data
   - Consider ensemble methods combining both approaches
   - Monitor for concept drift in fraud patterns over time
   - Consider cost-sensitive learning based on business impact
""")