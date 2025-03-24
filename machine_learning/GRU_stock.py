import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.font_manager as fm
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 字体文件路径（确保路径正确）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 创建字体对象
my_font = fm.FontProperties(fname=font_path)


# 获取单只股票数据
def load_stock_data(symbol):
    stock_df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq", period="daily")
    stock_df = stock_df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅', '换手率']]
    stock_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'pct_change', 'turnover_rate']
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.sort_values('date', inplace=True)
    return stock_df

# 获取沪深300所有股票数据并合并
def load_hs300_data():
    hs300_stocks = ak.index_stock_cons('000300')['品种代码']
    dfs = []
    for symbol in hs300_stocks:
        try:
            df = load_stock_data(symbol)
            df = add_technical_indicators(df)
            df['symbol'] = symbol
            print(f"加载股票{symbol}成功")
            dfs.append(df)
        except Exception as e:
            print(f"加载股票{symbol}失败: {e}")
    return pd.concat(dfs, ignore_index=True)

# 自定义技术指标计算（不使用talib）
def add_technical_indicators(df):
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

# 数据预处理
def preprocess_data(df, look_back=20):
    features = ['open', 'high', 'low', 'close', 'volume', 'pct_change', 'turnover_rate', 'MACD', 'MACD_signal', 'RSI', 'SMA']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back + 1, 3])

    return np.array(X), np.array(y), scaler

# 模型定义
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 主程序
if __name__ == '__main__':
    hs300_df = load_hs300_data()

    look_back = 30
    X, y, scaler = preprocess_data(hs300_df, look_back)

    model = build_model((look_back, X.shape[2]))

    model.fit(
        X, y, epochs=50, batch_size=128,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # 单独用601868验证
    stock_df = load_stock_data("601868")
    stock_df = add_technical_indicators(stock_df)
    X_stock, y_stock, scaler_stock = preprocess_data(stock_df, look_back)

    y_pred = model.predict(X_stock)

    def inverse_transform(scaler, y_data):
        dummy = np.zeros((len(y_data), scaler.n_features_in_))
        dummy[:, 3] = y_data.flatten()
        return scaler.inverse_transform(dummy)[:, 3]

    y_real = inverse_transform(scaler_stock, y_stock)
    y_pred_real = inverse_transform(scaler_stock, y_pred)

    mse = mean_squared_error(y_real, y_pred_real)
    mae = mean_absolute_error(y_real, y_pred_real)

    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")

    plt.figure(figsize=(14, 7))
    plt.plot(stock_df['date'][-len(y_real):], y_real, label='真实价格')
    plt.plot(stock_df['date'][-len(y_pred_real):], y_pred_real, label='预测价格')
    plt.legend(prop=my_font)
    plt.title('沪深300训练 - 单股票预测验证',fontproperties=my_font)
    plt.xlabel('日期',fontproperties=my_font)
    plt.ylabel('股价 (元)',fontproperties=my_font)
    plt.legend()
    plt.grid()
    plt.show()

    direction_accuracy = np.mean(np.sign(np.diff(y_real)) == np.sign(np.diff(y_pred_real))) * 100
    print(f"预测涨跌方向准确率: {direction_accuracy:.2f}%")
