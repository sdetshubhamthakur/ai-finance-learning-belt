import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download 5 years of MSFT daily data
ticker = 'MSFT'
data = yf.download(ticker, period='5y')
data.to_csv('MSFT_5yr.csv')   # Save for re-use

# Step 2: Load CSV into DataFrame
df = pd.read_csv('MSFT_5yr.csv', index_col='Date', parse_dates=True)

# Read data is loaded and number of rows
print("Data Loaded Successfully")
print(f"Number of rows: {df.shape[0]}")

# Step 3: Handle missing values
df.ffill(inplace=True)

# Step 4: Calculate daily percentage return
df['Daily Return'] = df['Close_price'].pct_change() * 100

# Step 5: Calculate 50-day & 200-day simple moving averages
df['SMA_50'] = df['Close_price'].rolling(window=50).mean()
df['SMA_200'] = df['Close_price'].rolling(window=200).mean()

# Step 6: Plot Closing Price and SMAs
#Identify "golden cross" and "death cross" and mark on plot
golden_crosses = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
death_crosses = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
plt.figure(figsize=(14,7))
plt.plot(df['Close_price'], label='Close Price')
plt.plot(df['SMA_50'], label='50-Day SMA')
plt.plot(df['SMA_200'], label='200-Day SMA')
plt.scatter(df.index[golden_crosses], df['Close_price'][golden_crosses], marker='^', color='g', s=100, label='Golden Cross')    
plt.scatter(df.index[death_crosses], df['Close_price'][death_crosses], marker='v', color='r', s=100, label='Death Cross')
plt.title('MSFT Closing Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()
plt.savefig('MSFT_Closing_Price_and_Moving_Averages.png')
plt.show()

# Step 7: Plot Daily Returns Histogram
plt.figure(figsize=(10,5))
plt.hist(df['Daily Return'].dropna(), bins=50, color='blue', alpha=0.7)
plt.title('MSFT Daily Returns Histogram')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.grid()
plt.savefig('MSFT_Daily_Returns_Histogram.png')
plt.show()
