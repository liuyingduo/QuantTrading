import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak 

stock_hfq_df = ak.stock_zh_a_hist(symbol="600237", adjust="hfq").iloc[:, :7]
df = stock_hfq_df

# Convert date column to datetime format
df['日期'] = pd.to_datetime(df['日期'])

# Calculate moving averages and technical indicators
df['MA5'] = df['收盘'].rolling(window=5).mean()
df['MA10'] = df['收盘'].rolling(window=10).mean()
df['MA50'] = df['收盘'].rolling(window=50).mean()
df['MA200'] = df['收盘'].rolling(window=200).mean()
df['MA_Volume_5D'] = df['成交量'].rolling(window=5).mean()

delta = df['收盘'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

short_ema = df['收盘'].ewm(span=12, adjust=False).mean()
long_ema = df['收盘'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Define short-term high/low price windows
lookback_period = 20
df['Min_Close_20D'] = df['收盘'].rolling(window=lookback_period).min()
df['Max_Close_20D'] = df['收盘'].rolling(window=lookback_period).max()

# Optimized short-term buy signal
df['Short_Buy_Signal'] = (
    (df['收盘'] <= df['Min_Close_20D']) &  # 20-day low
    (df['RSI'] < 30) &  # Oversold
    (df['收盘'] > df['MA5'] * 0.98) &  # Near or above MA5
    (df['成交量'] > df['MA_Volume_5D'] * 0.8)  # Volume confirmation
)

# Optimized short-term sell signal
df['Short_Sell_Signal'] = (
    (df['收盘'] >= df['Max_Close_20D']) &  # 20-day high
    (df['MACD'] < df['Signal_Line'])  # MACD dead cross
)

# Execute trading strategy
initial_cash = 100000
cash = initial_cash
shares = 0
trade_log = []

for index, row in df.iterrows():
    if row['Short_Buy_Signal'] and cash >= row['收盘']:
        shares = cash // row['收盘']
        cash -= shares * row['收盘']
        trade_log.append(('Buy', row['日期'], row['收盘'], shares))
    
    if row['Short_Sell_Signal'] and shares > 0:
        cash += shares * row['收盘']
        trade_log.append(('Sell', row['日期'], row['收盘'], shares))
        shares = 0

# Calculate final portfolio value
final_portfolio_value = cash + (shares * df.iloc[-1]['收盘'])
roi = (final_portfolio_value - initial_cash) / initial_cash * 100

# Plot stock price with buy/sell signals
plt.figure(figsize=(12,6))
plt.plot(df['日期'], df['收盘'], label="Closing Price", color='blue', alpha=0.6)
plt.scatter(df.loc[df['Short_Buy_Signal'], '日期'], df.loc[df['Short_Buy_Signal'], '收盘'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(df.loc[df['Short_Sell_Signal'], '日期'], df.loc[df['Short_Sell_Signal'], '收盘'], marker='v', color='red', label='Sell Signal', alpha=1)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price with Short-Term Buy & Sell Signals")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

print(f"Final ROI: {roi:.2f}%")
