import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime, timedelta

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fetch_northbound_data(start_date=None, end_date=None):
    """
    获取北向资金历史数据
    
    参数:
    start_date (str): 开始日期, 格式: 'YYYY-MM-DD'
    end_date (str): 结束日期, 格式: 'YYYY-MM-DD'
    
    返回:
    pd.DataFrame: 北向资金历史数据
    """
    print("获取北向资金历史数据...")
    # 使用akshare获取北向资金数据
    df = ak.stock_hsgt_hist_em(symbol="沪股通")
    print(df)
    
    # 转换日期格式
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 如果指定了日期范围，进行筛选
    if start_date:
        df = df[df['日期'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['日期'] <= pd.to_datetime(end_date)]
    
    # 按日期降序排序
    df = df.sort_values(by='日期', ascending=False)
    
    # 重置索引
    df = df.reset_index(drop=True)
    
    return df

def calculate_noise_factor(df, window=5):
    """
    计算北向噪声因子 = |北向资金净流入 / 5日北向资金流动均值|
    
    参数:
    df (pd.DataFrame): 北向资金数据
    window (int): 移动平均窗口大小，默认为5天
    
    返回:
    pd.DataFrame: 包含噪声因子的数据框
    """
    print(f"计算北向噪声因子，使用{window}日移动平均...")
    
    # 首先按日期升序排序，以便正确计算移动平均
    df = df.sort_values(by='日期')
    
    # 计算移动平均
    df['北向资金流动均值'] = df['当日成交净买额'].rolling(window=window).mean()
    
    # 计算噪声因子
    df['北向噪声因子'] = abs(df['当日成交净买额'] / df['北向资金流动均值'])
    
    # 再次按日期降序排序
    df = df.sort_values(by='日期', ascending=False)
    
    return df

def analyze_noise_factor(df, threshold=1.5):
    """
    分析北向噪声因子，找出异常值
    
    参数:
    df (pd.DataFrame): 包含噪声因子的数据框
    threshold (float): 噪声因子阈值，超过该值被视为异常
    
    返回:
    pd.DataFrame: 异常值数据框
    """
    print(f"分析北向噪声因子，阈值为{threshold}...")
    
    # 找出噪声因子超过阈值的记录
    anomaly_df = df[df['北向噪声因子'] > threshold].copy()
    
    # 添加分析结果
    anomaly_df['异常类型'] = np.where(anomaly_df['当日成交净买额'] > 0, '资金异常流入', '资金异常流出')
    
    return anomaly_df

def visualize_noise_factor(df, last_n_days=30):
    """
    可视化最近N天的北向噪声因子
    
    参数:
    df (pd.DataFrame): 包含噪声因子的数据框
    last_n_days (int): 最近天数，默认30天
    """
    print(f"生成最近{last_n_days}天的北向噪声因子可视化...")
    
    # 选择最近N天的数据
    recent_df = df.head(last_n_days).copy()
    
    # 按日期升序排序以便正确显示时间序列
    recent_df = recent_df.sort_values(by='日期')
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 绘制北向资金净流入
    ax1.plot(recent_df['日期'], recent_df['当日成交净买额'], 'b-', label='北向资金净流入(亿元)')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('北向资金净流入(亿元)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 在同一图表上绘制噪声因子
    ax2 = ax1.twinx()
    ax2.plot(recent_df['日期'], recent_df['北向噪声因子'], 'r-', label='北向噪声因子')
    ax2.set_ylabel('北向噪声因子', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 添加水平线表示阈值
    ax2.axhline(y=1.5, color='g', linestyle='--', alpha=0.7, label='异常阈值(1.5)')
    
    # 设置x轴标签旋转
    plt.xticks(rotation=45)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('北向资金流入与噪声因子分析')
    plt.tight_layout()
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存图表
    fig_path = os.path.join(results_dir, f"北向噪声因子_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(fig_path)
    print(f"图表已保存至: {fig_path}")
    
    # 显示图表
    plt.show()

def save_results(df, anomaly_df=None):
    """
    保存结果到CSV文件
    
    参数:
    df (pd.DataFrame): 完整数据框
    anomaly_df (pd.DataFrame): 异常值数据框
    """
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存完整数据
    full_path = os.path.join(results_dir, f"北向资金噪声因子_{datetime.now().strftime('%Y%m%d')}.csv")
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"完整数据已保存至: {full_path}")
    
    # 如果有异常值数据，也保存
    if anomaly_df is not None and not anomaly_df.empty:
        anomaly_path = os.path.join(results_dir, f"北向资金异常波动_{datetime.now().strftime('%Y%m%d')}.csv")
        anomaly_df.to_csv(anomaly_path, index=False, encoding='utf-8-sig')
        print(f"异常值数据已保存至: {anomaly_path}")

def main():
    """
    主函数
    """
    print("北向资金噪声因子分析工具")
    print("-" * 50)
    
    # 默认分析最近90天数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 获取数据
    df = fetch_northbound_data(start_date, end_date)
    print(f"成功获取{len(df)}条北向资金历史数据")
    
    # 计算噪声因子
    df_with_noise = calculate_noise_factor(df)
    
    # 分析异常值
    anomaly_df = analyze_noise_factor(df_with_noise)
    print(f"检测到{len(anomaly_df)}条异常数据")
    
    # 保存结果
    save_results(df_with_noise, anomaly_df)
    
    # 可视化最近30天数据
    visualize_noise_factor(df_with_noise, 30)
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()