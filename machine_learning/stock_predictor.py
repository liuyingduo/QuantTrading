import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import os

# 字体文件路径
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)

def load_stock_data(symbol):
    """获取单只股票数据"""
    stock_df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq", period="daily")
    stock_df = stock_df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅', '换手率']]
    stock_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'pct_change', 'turnover_rate']
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.sort_values('date', inplace=True)
    return stock_df

def add_technical_indicators(df):
    """添加技术指标"""
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA'] = df['close'].rolling(window=10).mean()
    df.bfill(inplace=True)
    return df

def generate_trading_signals(model, stock_data, scaler, look_back=30, confidence_threshold=0.02):
    """生成交易信号"""
    features = ['open', 'high', 'low', 'close', 'volume', 'pct_change', 'turnover_rate', 'MACD', 'MACD_signal', 'RSI', 'SMA']
    scaled_data = scaler.transform(stock_data[features])
    
    X = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
    X = np.array(X)
    
    y_pred = model.predict(X)
    
    def inverse_transform(scaler, y_data):
        dummy = np.zeros((len(y_data), scaler.n_features_in_))
        dummy[:, 3] = y_data.flatten()
        return scaler.inverse_transform(dummy)[:, 3]
    
    y_pred_real = inverse_transform(scaler, y_pred)
    
    current_prices = stock_data['close'].iloc[look_back:].values
    predicted_changes = (y_pred_real - current_prices) / current_prices
    
    signals = pd.DataFrame({
        'date': stock_data['date'].iloc[look_back:],
        'current_price': current_prices,
        'predicted_price': y_pred_real,
        'predicted_change': predicted_changes,
        'signal': 0
    })
    
    signals.loc[signals['predicted_change'] > confidence_threshold, 'signal'] = 1
    signals.loc[signals['predicted_change'] < -confidence_threshold, 'signal'] = -1
    
    return signals

def evaluate_trading_strategy(signals, initial_capital=100000):
    """评估交易策略"""
    position = 0
    capital = initial_capital
    trades = []
    
    for i in range(len(signals)):
        if signals['signal'].iloc[i] == 1 and position == 0:
            shares = capital // signals['current_price'].iloc[i]
            cost = shares * signals['current_price'].iloc[i]
            capital -= cost
            position = shares
            trades.append({
                'date': signals['date'].iloc[i],
                'type': 'buy',
                'price': signals['current_price'].iloc[i],
                'shares': shares,
                'capital': capital
            })
        elif signals['signal'].iloc[i] == -1 and position > 0:
            revenue = position * signals['current_price'].iloc[i]
            capital += revenue
            trades.append({
                'date': signals['date'].iloc[i],
                'type': 'sell',
                'price': signals['current_price'].iloc[i],
                'shares': position,
                'capital': capital
            })
            position = 0
    
    final_capital = capital + (position * signals['current_price'].iloc[-1] if position > 0 else 0)
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    return {
        'trades': trades,
        'final_capital': final_capital,
        'total_return': total_return,
        'num_trades': len(trades)
    }

def main():
    # 加载保存的模型和参数
    model_dir = 'saved_models'
    model = load_model(os.path.join(model_dir, 'best_model.h5'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    model_params = joblib.load(os.path.join(model_dir, 'model_params.pkl'))
    
    print("模型加载成功！")
    
    # 获取最新的沪深300成分股
    latest_stocks = ak.index_stock_cons('000300')['品种代码']
    
    # 存储所有股票的交易信号
    all_signals = {}
    
    for symbol in latest_stocks:
        try:
            # 获取股票数据
            stock_data = load_stock_data(symbol)
            stock_data = add_technical_indicators(stock_data)
            
            # 生成交易信号
            signals = generate_trading_signals(model, stock_data, scaler, model_params['look_back'])
            
            # 评估交易策略
            strategy_results = evaluate_trading_strategy(signals)
            
            # 只保存最近一天的信号
            latest_signal = signals.iloc[-1]
            
            all_signals[symbol] = {
                'signal': latest_signal['signal'],
                'predicted_change': latest_signal['predicted_change'],
                'current_price': latest_signal['current_price'],
                'predicted_price': latest_signal['predicted_price'],
                'strategy_return': strategy_results['total_return']
            }
            
        except Exception as e:
            print(f"处理股票{symbol}时出错: {e}")
    
    # 按预测涨跌幅排序，选择前10只股票
    sorted_stocks = sorted(all_signals.items(), 
                         key=lambda x: x[1]['predicted_change'], 
                         reverse=True)
    
    print("\n推荐买入的股票（按预测涨跌幅排序）:")
    print("股票代码\t当前价格\t预测价格\t预测涨跌幅\t策略收益率")
    print("-" * 60)
    for symbol, info in sorted_stocks[:10]:
        if info['signal'] == 1:
            print(f"{symbol}\t{info['current_price']:.2f}\t{info['predicted_price']:.2f}\t"
                  f"{info['predicted_change']*100:.2f}%\t{info['strategy_return']:.2f}%")
    
    print("\n推荐卖出的股票（按预测跌幅排序）:")
    print("股票代码\t当前价格\t预测价格\t预测涨跌幅\t策略收益率")
    print("-" * 60)
    for symbol, info in sorted(sorted_stocks, key=lambda x: x[1]['predicted_change'])[:10]:
        if info['signal'] == -1:
            print(f"{symbol}\t{info['current_price']:.2f}\t{info['predicted_price']:.2f}\t"
                  f"{info['predicted_change']*100:.2f}%\t{info['strategy_return']:.2f}%")

if __name__ == '__main__':
    main()