import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# =========== 1. 解决中文显示问题 ===========
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# =========== 2. 准备数据 ===========
# 2.1 使用真实数据：
# data = pd.read_csv('your_stock_data.csv', parse_dates=['日期'])
# data.set_index('日期', inplace=True)

# 2.2 使用随机游走模拟的股票数据：
np.random.seed(42)
data_size = 200
price_changes = np.random.normal(loc=0, scale=1, size=data_size) 
price = 100 + np.cumsum(price_changes)  # 从初始价格 100 开始，累加随机波动
date_range = pd.date_range(start='2022-01-01', periods=data_size, freq='D')

data = pd.DataFrame({
    '日期': date_range,
    '收盘价': price
})
data.set_index('日期', inplace=True)

# =========== 3. 设置双移动平均窗口并计算 ===========
short_window = 5    # 短期移动平均窗口
long_window = 20    # 长期移动平均窗口

data['短期MA'] = data['收盘价'].rolling(window=short_window).mean()
data['长期MA'] = data['收盘价'].rolling(window=long_window).mean()

# =========== 4. 生成买入、卖出信号 ===========
data['信号'] = 0  # 默认无信号
# 从第short_window行开始比较，否则前面没有足够的MA数据
data['信号'][long_window:] = np.where(
    data['短期MA'][long_window:] > data['长期MA'][long_window:], 
    1, 
    0
)
# 买入信号：今天信号与昨天信号之差为 1
data['买入'] = data['信号'].diff() == 1
# 卖出信号：今天信号与昨天信号之差为 -1
data['卖出'] = data['信号'].diff() == -1

# =========== 5. 可视化 =========== 
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['收盘价'], label='收盘价', color='gray', alpha=0.6)
plt.plot(data.index, data['短期MA'], label=f'{short_window}日短期MA', color='blue')
plt.plot(data.index, data['长期MA'], label=f'{long_window}日长期MA', color='red')

# 标记买入和卖出位置
plt.scatter(data.index[data['买入']], data['收盘价'][data['买入']], marker='^', color='g', label='买入信号', s=100)
plt.scatter(data.index[data['卖出']], data['收盘价'][data['卖出']], marker='v', color='r', label='卖出信号', s=100)

plt.title('双移动平均策略示例')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.show()
