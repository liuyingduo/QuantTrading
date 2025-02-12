import akshare as ak
import pandas as pd
import numpy as np
import traceback
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
import time
import json
import os
from datetime import datetime
import concurrent.futures

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_fund_portfolio(fund_code, date):
    """Get fund portfolio with caching"""
    cache_file = os.path.join(CACHE_DIR, f"{fund_code}_{date}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Add delay to prevent rate limiting
    time.sleep(1)
    
    # Try API call with retries
    try:
        portfolio_df = ak.fund_portfolio_hold_em(symbol=fund_code, date=date)
        stocks = portfolio_df["股票代码"].dropna().unique().tolist()
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(stocks, f)
        
        return stocks
        
    except Exception as e:
        print(f"获取基金 {fund_code} 持仓数据失败：", e)
        return []

def process_funds(selected_funds):
    fund_stock_dict = {}
    current_year = 2024
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        future_to_fund = {
            executor.submit(get_fund_portfolio, fund_code, str(current_year)): fund_code 
            for fund_code in selected_funds["基金代码"]
        }
        
        for future in concurrent.futures.as_completed(future_to_fund):
            fund_code = future_to_fund[future]
            try:
                stocks = future.result()
                fund_stock_dict[fund_code] = stocks
            except Exception as e:
                print(f"处理基金 {fund_code} 失败：", e)
                fund_stock_dict[fund_code] = []
    
    return fund_stock_dict

# =======================
# 1. 获取基金排行数据，并过滤出各期均为正收益的基金
# =======================
fund_ranking_df = ak.fund_open_fund_rank_em(symbol="全部")
# 查看数据列，感兴趣的列有："日增长率", "近1周", "近1月", "近3月", "近6月"
# 有时这些字段可能带有 "%" 字符，因此需要清洗数据
def clean_pct(x):
    try:
        return float(str(x).replace('%', ''))
    except Exception:
        return np.nan

for col in ["日增长率", "近1周", "近1月", "近3月", "近6月"]:
    fund_ranking_df[col] = fund_ranking_df[col].apply(clean_pct)

# 选取这5个指标均大于0的基金（即各周期均为正增长） #    
selected_funds = fund_ranking_df[
    (fund_ranking_df["日增长率"] > 0) &
    (fund_ranking_df["近1周"] > 0) &
    (fund_ranking_df["近1月"] > 0) &
    (fund_ranking_df["近3月"] > 0) &
    (fund_ranking_df["近6月"] > 0)
].copy()

print("满足条件的基金数量：", len(selected_funds))

# 选取前 10 只基金进行后续分析
# selected_funds = selected_funds.head(10)
print("选取的基金：",selected_funds[["基金代码", "基金简称"]])
selected_funds.to_excel("selected_funds.xlsx", index=False)

# =======================
# 2. 获取每只基金的股票持仓数据
# =======================
# 注意：此处调用的是 ak.fund_portfolio_hold_em 接口，输入参数 date 为指定年份（如"2024"）
fund_stock_dict = process_funds(selected_funds)

# =======================
# 3. 根据各基金持仓的股票构造二值特征矩阵，便于后续聚类
# =======================
# 收集所有出现过的股票代码
all_stocks = []
valid_fund_codes = []  # 保存成功获取数据的基金代码
for fund_code, stocks in fund_stock_dict.items():
    if stocks:  # 如果该基金有成功获取持仓数据
        all_stocks.extend(stocks)
        valid_fund_codes.append(fund_code)

# 去除重复的股票代码
all_stocks = list(set(all_stocks))


# 利用 MultiLabelBinarizer 将每个基金的持仓转化为一个二值向量（全市场股票列表作为特征）
mlb = MultiLabelBinarizer(classes=all_stocks)
# 只使用成功获取数据的基金代码
holdings_list = [fund_stock_dict.get(fund, []) for fund in valid_fund_codes]
X = mlb.fit_transform(holdings_list)

# =======================
# 4. 利用聚类方法对基金进行归类（持仓相似的归为一类），修改为仅对有效基金进行聚类
# 聚类的数量可根据实际数据进行调整，这里设定为最多分成 5 类（若基金数少则取较小值）
n_clusters = min(5, len(valid_fund_codes))
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
clusters = clustering.fit_predict(X)

# 将聚类结果添加到 selected_funds 数据框中
selected_funds["cluster"] = selected_funds["基金代码"].apply(
    lambda x: clusters[valid_fund_codes.index(x)] if x in valid_fund_codes else -1  # 无数据的基金标记为-1
)


# 处理所有基金，确保那些没有成功聚类的基金作为一类来计算后续指标
fund_stock_dict_with_failed = {}
for fund_code, stocks in fund_stock_dict.items():
    if fund_code in selected_funds[selected_funds["cluster"] == -1]["基金代码"].values:
        # 如果基金未能成功聚类，将其作为一个特殊类别的基金处理
        fund_stock_dict_with_failed[fund_code] = stocks
    else:
        fund_stock_dict_with_failed[fund_code] = stocks

# =======================
# 6. 对每个选中的基金，调用历史净值数据接口并计算各项指标
#     指标包括：夏普比率、波动率、风险收益指数（此处简单定义为收益均值/波动率）
# =======================
def compute_metrics(fund_code):
    try:
        # 获取历史净值数据（单位净值走势）
        hist_df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        # 按净值日期排序
        hist_df = hist_df.sort_values("净值日期")
        # 计算每日收益率（单位净值的百分比变化）
        hist_df["return"] = hist_df["单位净值"].pct_change()
        returns = hist_df["return"].dropna()
        if returns.empty:
            return np.nan, np.nan, np.nan
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else np.nan
        # 波动率：使用每日收益率标准差
        volatility = returns.std()
        # 风险收益指数：此处简单定义为收益均值与波动率的比值
        risk_return = returns.mean() / volatility if volatility != 0 else np.nan
        return sharpe_ratio, volatility, risk_return
    except Exception as e:
        print(f"基金 {fund_code} 计算指标出错：", e)
        return np.nan, np.nan, np.nan

# 计算指标并处理聚类失败的基金
# 5. 计算每个选中基金的夏普比率、波动率等指标，排除没有持仓数据的基金
metrics_list = []
for idx, row in selected_funds.iterrows():
    fund_code = row["基金代码"]
    if fund_code in valid_fund_codes:  # 只处理有数据的基金
        sharpe, vol, rr = compute_metrics(fund_code)
        metrics_list.append({
            "基金代码": fund_code,
            "夏普比率": sharpe,
            "波动率": vol,
            "风险收益指数": rr
        })

metrics_df = pd.DataFrame(metrics_list)
print("各选中基金的绩效指标：")
print(metrics_df)

metrics_df.to_excel("fund_metrics.xlsx", index=False)

# =======================
# 7. 依据指标与权重计算综合得分，选出最终的基金
#     此处示例构造综合得分公式：
#         综合得分 = 6 * 夏普比率 - 4 * 波动率 + 1 * 风险收益指数
#     （即追求夏普比率越高越好、波动率越低越好、风险收益指数越大越好）
# =======================
metrics_df["composite_score"] = 6 * metrics_df["夏普比率"] - 4 * metrics_df["波动率"] + metrics_df["风险收益指数"]
final_fund = metrics_df.sort_values(by="composite_score", ascending=False).iloc[0]

print("最终选出的基金为：")
print(final_fund)
