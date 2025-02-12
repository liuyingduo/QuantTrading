
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示问题

# ========== 获取基金历史净值数据 ==========

def get_fund_data(fund_code):
    headers = {
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Connection': 'keep-alive',
    'Cookie': 'qgqp_b_id=a12af75fc3a87ca21ba8acb708d03110; websitepoptg_api_time=1738494622931; st_si=86284209976786; st_asi=delete; fund_registerAd_1=1; FundWebTradeUserInfo=JTdCJTIyQ3VzdG9tZXJObyUyMjolMjIlMjIsJTIyQ3VzdG9tZXJOYW1lJTIyOiUyMiUyMiwlMjJWaXBMZXZlbCUyMjolMjIlMjIsJTIyTFRva2VuJTIyOiUyMiUyMiwlMjJJc1Zpc2l0b3IlMjI6JTIyJTIyLCUyMlJpc2slMjI6JTIyJTIyLCUyMlN1cnZleURheSUyMjowLCUyMklzQXVkaXROZWVkUG9wJTIyOnRydWUlN0Q%3D; fullscreengg=1; fullscreengg2=1; EMFUND0=null; EMFUND1=02-02%2019%3A11%3A14@%23%24%u9E4F%u534E%u78B3%u4E2D%u548C%u4E3B%u9898%u6DF7%u5408C@%23%24016531; EMFUND3=02-02%2019%3A12%3A50@%23%24%u4E2D%u822A%u4F18%u9009%u9886%u822A%u6DF7%u5408%u53D1%u8D77C@%23%24022853; EMFUND4=02-02%2019%3A24%3A29@%23%24%u91D1%u5143%u987A%u5B89%u5143%u542F%u7075%u6D3B%u914D%u7F6E%u6DF7%u5408@%23%24004685; EMFUND5=02-02%2019%3A24%3A55@%23%24%u5BCC%u56FD%u65B0%u6750%u6599%u65B0%u80FD%u6E90%u6DF7%u5408A@%23%24009092; EMFUND7=02-02%2019%3A26%3A25@%23%24%u957F%u57CE%u4E45%u946B%u6DF7%u5408A@%23%24000649; EMFUND8=02-02%2019%3A27%3A34@%23%24%u4EA4%u94F6%u79D1%u6280%u521B%u65B0%u7075%u6D3B%u914D%u7F6E%u6DF7%u5408A@%23%24519767; EMFUND6=02-02%2019%3A45%3A37@%23%24%u524D%u6D77%u5F00%u6E90%u5609%u946B%u6DF7%u5408A@%23%24001765; EMFUND9=02-02%2019%3A48%3A59@%23%24%u5BCC%u56FD%u5929%u745E%u5F3A%u52BF%u6DF7%u5408@%23%24100022; EMFUND2=02-02 19:55:52@#$%u4E2D%u822A%u8D8B%u52BF%u9886%u822A%u6DF7%u5408%u53D1%u8D77A@%23%24021489; st_pvi=03866456682752; st_sp=2025-02-02%2019%3A10%3A24; st_inirUrl=https%3A%2F%2Fwww.bing.com%2F; st_sn=21; st_psi=20250202195603700-112200305283-8446184315',
    'Referer': 'https://fundf10.eastmoney.com/',
    'Sec-Fetch-Dest': 'script',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0',
    'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Microsoft Edge";v="132"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"'
    }
    data_list = []
    for page in range(1, 300):
        url = f"https://api.fund.eastmoney.com/f10/lsjz?callback=jQuery183019642445978256595_1738497363766&fundCode={fund_code}&pageIndex={page}&pageSize=20&startDate=&endDate=&_=1738497397711"

        response = requests.get(url, headers=headers)
        data = response.text
        
        # 提取 JSON 部分
        json_str = data.split('(', 1)[1].rsplit(')', 1)[0]
        data = json.loads(json_str)

        # 获取历史净值数据
        lsjz_list = data["Data"]["LSJZList"]
        if not lsjz_list:
            break

        for item in lsjz_list:
            data_list.append([item["FSRQ"], float(item["DWJZ"]), float(item["LJJZ"]), float(item["JZZZL"]) if item["JZZZL"] and item["JZZZL"] != "" else 0])

    # 转换为 DataFrame
    df = pd.DataFrame(data_list, columns=["日期", "单位净值", "累计净值", "净值增长率"])
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values(by="日期")  # 按日期排序
    return df

# ========== 计算关键财务指标 ==========
def calculate_fund_metrics(df):
    expected_return = df["净值增长率"].mean() / 100  # 转换为小数
    std_dev_return = df["净值增长率"].std() / 100  # 转换为小数
    annualized_return = ((1 + expected_return) ** 252) - 1
    risk_free_rate = 0.02  # 设定无风险收益率 2%
    sharpe_ratio = (annualized_return - risk_free_rate) / (std_dev_return * np.sqrt(252))

    return {
        "平均每日回报率（数学期望）": expected_return,
        "每日回报率标准差": std_dev_return,
        "年化收益率": annualized_return,
        "夏普比率": sharpe_ratio
    }

# ========== 预测未来 30 天净值 ==========
def predict_future(df, expected_return):
    future_days = 30
    future_dates = pd.date_range(start=df["日期"].max(), periods=future_days+1, freq='D')[1:]
    predicted_values = [df["单位净值"].iloc[-1] * (1 + expected_return) ** i for i in range(future_days)]
    return future_dates, predicted_values

# ========== 可视化基金趋势 ==========
def plot_fund_trend(df, future_dates, predicted_values, fund_code):
    plt.figure(figsize=(10, 5))
    plt.plot(df["日期"], df["单位净值"], label="历史单位净值", color="blue")
    plt.plot(future_dates, predicted_values, label="未来预测净值", linestyle="dashed", color="red")
    plt.xlabel("日期")
    plt.ylabel("单位净值")
    plt.title(f"基金 {fund_code} 单位净值趋势及预测")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# ========== 主函数 ==========
def main():
    fund_code = input("请输入基金代码（如 012921）：").strip()
    print(f"正在获取基金 {fund_code} 数据...")
    
    df = get_fund_data(fund_code)
    df.to_excel(f"{fund_code}_fund_data.xlsx", index=False)
    
    print(f"基金 {fund_code} 数据已保存为 {fund_code}_fund_data.xlsx")

    # 计算基金指标
    metrics = calculate_fund_metrics(df)
    print("\n基金关键财务指标：")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 预测未来 30 天净值
    future_dates, predicted_values = predict_future(df, metrics["平均每日回报率（数学期望）"])
    
    # 绘制趋势图
    plot_fund_trend(df, future_dates, predicted_values, fund_code)

if __name__ == "__main__":
    main()
