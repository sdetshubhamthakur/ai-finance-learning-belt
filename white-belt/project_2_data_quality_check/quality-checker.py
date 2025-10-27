import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('synthetic_financial_data.csv', parse_dates=['Date'])

# Flag missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Method 1: Previous Window Only (No Self-Inclusion)
def detect_outliers_external_window(series, window=50, threshold=4.5):
    """
    Detect outliers using Z-score with previous window only (no self-inclusion)
    """
    outliers = []
    for i in range(window, len(series)):
        # Use only previous values for statistics
        window_data = series.iloc[i-window:i]
        mean_val = window_data.mean()
        std_val = window_data.std()
        
        if std_val > 0:
            z_score = (series.iloc[i] - mean_val) / std_val
            outliers.append(abs(z_score) > threshold)
        else:
            outliers.append(False)
    
    # Pad with False for first 'window' values
    result = [False] * window + outliers
    return pd.Series(result, index=series.index)

# Method 1: External Window
external_outliers = detect_outliers_external_window(df['Close'], window=50, threshold=4.5)
print(f"External Window Method: {external_outliers.sum()} outliers detected")
# print details for outliers
outlier_details = df[external_outliers]
print("Outlier details:")
print(outlier_details)

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.scatter(df['Date'][external_outliers], df['Close'][external_outliers], color='red', label='Detected Outliers')
plt.title('Close Prices with Detected Outliers (External Window Method)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('outliers_external_window.png')
plt.grid()

# Suggest cleaning actions like further investigation using simple heuristics.
if external_outliers.sum() > 0:
    print("Suggested actions for detected outliers:")
    for index, row in outlier_details.iterrows():
        print(f" - For date {row['Date']}: consider investigating the cause of the outlier.")

plt.tight_layout()
plt.show()

