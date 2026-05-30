import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# Load Dataset
df = pd.read_csv("Stock Prices Data Set.csv")
# Filter Apple Stock Only
df = df[df['symbol'] == 'AAPL']
# Convert Date Column to Datetime
df['date'] = pd.to_datetime(df['date'])
# Sort by Date
df = df.sort_values('date')
# Set Date as Index
df.set_index('date', inplace=True)
# 1. Trend Analysis
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'])
plt.title("Stock Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.savefig("Trend.png")
plt.show()
# 2. Moving Average Analysis
df['MA30'] = df['close'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Close Price')
plt.plot(df.index, df['MA30'], label='30-Day Moving Average')
plt.title("Moving Average Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("MovingAverage.png")
plt.show()
# 3. Seasonal Decomposition
result = seasonal_decompose(
    df['close'],
    model='additive',
    period=30
)
fig = result.plot()
fig.set_size_inches(12, 8)
fig.suptitle(
    "Seasonal Decomposition of AAPL Stock Prices",
    fontsize=16
)
plt.savefig("Decomposition.png")
plt.show()
print("Time Series Analysis Completed Successfully!")