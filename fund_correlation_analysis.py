import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import concurrent.futures
import os
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# 添加GPU加速相关的导入
try:
    import cupy as cp
    USE_GPU = True
    print("GPU加速已启用")
except ImportError:
    USE_GPU = False
    print("未检测到GPU加速库，将使用CPU模式运行")

# 创建缓存目录
CACHE_DIR = Path("./fund_data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(fund_code, start_date, end_date):
    """获取缓存文件路径"""
    cache_key = f"{fund_code}_{start_date}_{end_date}"
    return CACHE_DIR / f"{cache_key}.pkl"

def load_from_cache(cache_path):
    """从缓存加载数据"""
    try:
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            # 检查缓存是否过期（超过1天）
            if datetime.now().timestamp() - os.path.getmtime(cache_path) < 86400:
                return data
    except Exception:
        pass
    return None

def save_to_cache(cache_path, data):
    """保存数据到缓存"""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"缓存保存失败: {e}")

def get_fund_daily_data(fund_code, start_date=None, end_date=None):
    """获取基金的历史净值数据"""
    try:
        # 检查缓存
        cache_path = get_cache_path(fund_code, start_date, end_date)
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data

        # 获取历史净值数据
        hist_df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        
        # 按净值日期排序
        hist_df = hist_df.sort_values("净值日期")
        hist_df.set_index("净值日期", inplace=True)
        
        # 如果指定了日期范围，进行过滤
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            hist_df.index = pd.to_datetime(hist_df.index)
            hist_df = hist_df[(hist_df.index >= start_date) & (hist_df.index <= end_date)]
            
        # 计算日收益率
        hist_df[f"{fund_code}_return"] = hist_df["单位净值"].pct_change()
        result = hist_df[[f"{fund_code}_return"]]
        
        # 保存到缓存
        save_to_cache(cache_path, result)
        
        return result
    except Exception as e:
        print(f"获取基金 {fund_code} 数据失败：", e)
        return None

def get_fund_data_parallel(fund_codes, start_date=None, end_date=None):
    """并行获取多个基金的数据"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 创建任务
        future_to_fund = {
            executor.submit(get_fund_daily_data, code, start_date, end_date): code 
            for code in fund_codes
        }
        
        # 收集结果
        all_returns = pd.DataFrame()
        for future in concurrent.futures.as_completed(future_to_fund):
            fund_code = future_to_fund[future]
            try:
                fund_data = future.result()
                if fund_data is not None:
                    all_returns = pd.concat([all_returns, fund_data], axis=1)
            except Exception as e:
                print(f"处理基金 {fund_code} 时发生错误：{e}")
                
        return all_returns

def calculate_correlation_matrix_gpu(data):
    """使用GPU计算相关性矩阵"""
    # 将数据转换为GPU数组
    gpu_data = cp.array(data.values)
    
    # 标准化数据
    means = cp.mean(gpu_data, axis=0)
    stds = cp.std(gpu_data, axis=0)
    normalized_data = (gpu_data - means) / stds
    
    # 计算相关性矩阵
    n = normalized_data.shape[1]
    corr_matrix = cp.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            corr = cp.mean(normalized_data[:, i] * normalized_data[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # 将结果转回CPU
    return pd.DataFrame(
        cp.asnumpy(corr_matrix), 
        index=data.columns, 
        columns=data.columns
    )

def analyze_fund_correlations(fund_codes, start_date=None, end_date=None):
    """分析多个基金之间的相关性"""
    if not start_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # 并行获取所有基金的收益率数据
    all_returns = get_fund_data_parallel(fund_codes, start_date, end_date)
    
    # 计算相关性矩阵
    if USE_GPU:
        correlation_matrix = calculate_correlation_matrix_gpu(all_returns)
    else:
        correlation_matrix = all_returns.corr()
    
    # 找出强负相关的基金对
    negative_pairs = []
    for i in range(len(fund_codes)):
        for j in range(i+1, len(fund_codes)):
            corr = correlation_matrix.iloc[i, j]
            if corr < -0.3:
                negative_pairs.append({
                    'fund1': fund_codes[i],
                    'fund2': fund_codes[j],
                    'correlation': corr
                })
    
    return correlation_matrix, negative_pairs, all_returns

def plot_correlation_heatmap(correlation_matrix, fund_codes):
    """绘制相关性热图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdYlBu', 
                center=0,
                xticklabels=fund_codes,
                yticklabels=fund_codes)
    plt.title('基金收益率相关性热图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_scatter_matrix(returns_data, fund_codes):
    """绘制散点图矩阵"""
    plt.figure(figsize=(15, 15))
    pd.plotting.scatter_matrix(returns_data, figsize=(15, 15))
    plt.tight_layout()
    plt.savefig('scatter_matrix.png')
    plt.close()

def get_selected_funds(top_n=10):
    """获取基金排行数据并筛选"""
    try:
        # 获取基金排行数据
        fund_ranking_df = ak.fund_open_fund_rank_em(symbol="全部")
        
        return fund_ranking_df
    except Exception as e:
        print("获取基金排行数据失败：", e)
        return None

def main():
    try:
        # 获取基金排行并筛选
        print("正在获取基金排行数据...")
        selected_funds = get_selected_funds(top_n=10)
        if selected_funds is None or len(selected_funds) == 0:
            print("未获取到符合条件的基金数据")
            return
            
        print(f"\n筛选出的基金数量：{len(selected_funds)}")
        print("\n选取的基金：")
        print(selected_funds[["基金代码", "基金简称", "近6月"]].to_string())
        
        fund_codes = selected_funds["基金代码"].tolist()
        
        # 设置时间范围（最近一年）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        print("\n正在进行相关性分析...")
        # 进行相关性分析
        correlation_matrix, negative_pairs, returns_data = analyze_fund_correlations(
            fund_codes, start_date, end_date
        )
        
        # 输出分析结果
        if negative_pairs:
            print("\n发现以下强负相关的基金对：")
            for pair in negative_pairs:
                fund1_name = selected_funds[selected_funds["基金代码"] == pair['fund1']]["基金简称"].iloc[0]
                fund2_name = selected_funds[selected_funds["基金代码"] == pair['fund2']]["基金简称"].iloc[0]
                print(f"{fund1_name}({pair['fund1']}) 和 {fund2_name}({pair['fund2']}) 的相关系数为: {pair['correlation']:.3f}")
        else:
            print("\n未发现强负相关的基金对")
        
        # 绘制相关性热图
        plot_correlation_heatmap(correlation_matrix, fund_codes)
        print("\n已生成相关性热图：correlation_heatmap.png")
        
        # 绘制散点图矩阵
        plot_scatter_matrix(returns_data, fund_codes)
        print("已生成散点图矩阵：scatter_matrix.png")
        
        # 保存相关性矩阵到Excel
        correlation_matrix.to_excel("fund_correlation_matrix.xlsx")
        print("相关性矩阵已保存到：fund_correlation_matrix.xlsx")
        
    except Exception as e:
        print("分析过程中出现错误：", e)
        print("详细错误信息：")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 