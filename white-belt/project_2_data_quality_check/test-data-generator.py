import numpy as np
import pandas as pd

# Create 1250 business days (approx 5 years)
dates = pd.bdate_range(start='2019-01-01', periods=1250)

np.random.seed(42)

# Simulate OHLCV data with a random walk for price
open_prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
close_prices = open_prices + np.random.normal(0, 2, len(dates))
high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 0.5, len(dates)))
low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 0.5, len(dates)))
volumes = np.random.randint(90000, 120000, len(dates)).astype(float)

# Intentionally add missing values
missing_indices = np.random.choice(len(dates), size=20, replace=False)
for i in missing_indices:
    close_prices[i] = np.nan

missing_indices_vol = np.random.choice(len(dates), size=20, replace=False)
for i in missing_indices_vol:
    volumes[i] = np.nan

# Inject outliers (very high and very low)
outlier_indices_close = np.random.choice(len(dates), size=5, replace=False)
close_prices[outlier_indices_close] += np.random.choice([50, -50], size=5)

outlier_indices_vol = np.random.choice(len(dates), size=5, replace=False)
volumes[outlier_indices_vol] *= np.random.choice([0.1, 5], size=5)

# Assemble DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volumes
})

df.to_csv('synthetic_financial_data.csv', index=False)
print("CSV 'synthetic_financial_data.csv' generated.")
