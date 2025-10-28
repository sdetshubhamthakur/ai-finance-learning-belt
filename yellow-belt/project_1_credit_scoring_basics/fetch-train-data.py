from ucimlrepo import fetch_ucirepo
import pandas as pd
  
# fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = pd.DataFrame(statlog_german_credit_data.data.features)
y = statlog_german_credit_data.data.targets.squeeze()

# Save features and targets as separate CSV files
X.to_csv('german_credit_features.csv', index=False)
y.to_csv('german_credit_targets.csv', index=False)

# Or combine features and targets into one CSV file
combined_data = X.copy()
combined_data['target'] = y
combined_data.to_csv('german_credit_data.csv', index=False)

print(f"Data saved successfully!")
print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# # metadata
print(statlog_german_credit_data.metadata)

# # variable information
print(statlog_german_credit_data.variables)
