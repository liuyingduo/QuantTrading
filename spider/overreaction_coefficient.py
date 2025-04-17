import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_financing_interference(symbol):
    """计算融资干扰因子 = 行业融资买入均值 / 个股融资买入额"""
    # 加载个股数据
    data_path = ".\\data\\rzrq_data_2025-04-16.json"
    with open(data_path, "r", encoding="utf-8") as f:
        stock_rzrq_list = json.load(f)
    
    # 加载行业数据
    sector_path = ".\\data\\rzrq_sector_data.json"
    with open(sector_path, "r", encoding="utf-8") as f:
        sector_rzrq_list = json.load(f)
    
    # 加载个股所属行业数据
    board_path = ".\\data\\stock_board.json"
    with open(board_path, "r", encoding="utf-8") as f:
        stock_board_list = json.load(f)
    
    # 1. 获取个股数据
    stock_info = None
    for item in stock_rzrq_list:
        if item.get("SCODE") == symbol:
            stock_info = item
            break
    
    if not stock_info:
        print(f"未找到股票代码为 {symbol} 的融资融券数据")
        return None
    
    # 2. 获取个股所属行业
    board_info = None
    for item in stock_board_list:
        if item.get("代码") == symbol:
            board_info = item
            break
    
    if not board_info:
        print(f"未找到股票代码为 {symbol} 的行业信息")
        return None
    
    sector_name = board_info.get("板块名称")
    sector_code = board_info.get("板块代码")
    print(sector_name, sector_code)
    
    # 3. 获取行业数据
    sector_info = None
    for item in sector_rzrq_list:
        if len(item.get("BOARD_CODE"))==3:
            BOARD_CODE = "BK0" + item.get("BOARD_CODE")
        else:
            BOARD_CODE = "BK" + item.get("BOARD_CODE")
        if BOARD_CODE == sector_code:
            sector_info = item
            break
    
    if not sector_info:
        print(f"未找到行业代码为 {sector_code} 的行业融资融券数据")
        return None
    
    # 4. 计算融资干扰因子
    stock_fin_buy_amt = stock_info.get("RZMRE", 0)  # 个股融资买入额
    sector_fin_buy_amt = sector_info.get("FIN_BUY_AMT", 0)  # 行业总融资买入额
    
    # 计算行业内股票数量 (行业总融资买入额除以个股平均融资买入额)
    sector_stock_count = 0
    for item in stock_rzrq_list:
        stock_board = None
        for board in stock_board_list:
            if board.get("代码") == item.get("SCODE"):
                stock_board = board
                break
        
        if stock_board and stock_board.get("板块代码") == sector_code:
            sector_stock_count += 1
    
    # 计算行业平均融资买入额
    if sector_stock_count > 0:
        sector_avg_fin_buy_amt = sector_fin_buy_amt / sector_stock_count
    else:
        sector_avg_fin_buy_amt = sector_fin_buy_amt
    
    # 计算融资干扰因子
    if stock_fin_buy_amt == 0:
        interference_factor = float('inf')  # 如果个股融资买入额为0，干扰因子为无穷大
    else:
        interference_factor = sector_avg_fin_buy_amt / stock_fin_buy_amt
    
    # 返回结果字典
    result = {
        "股票代码": symbol,
        "股票名称": stock_info.get("SECNAME", ""),
        "所属行业": sector_name,
        "个股融资买入额": stock_fin_buy_amt,
        "行业总融资买入额": sector_fin_buy_amt,
        "行业内股票数量": sector_stock_count,
        "行业平均融资买入额": sector_avg_fin_buy_amt,
        "融资干扰因子": interference_factor
    }
    
    return result

# 测试代码
if __name__ == "__main__":
    symbol = input("请输入股票代码(如600206): ")
    result = calculate_financing_interference(symbol)
    
    if result:
        print("\n融资干扰因子计算结果:")
        print(f"股票代码: {result['股票代码']}")
        print(f"股票名称: {result['股票名称']}")
        print(f"所属行业: {result['所属行业']}")
        print(f"个股融资买入额: {result['个股融资买入额']:,.2f}")
        print(f"行业总融资买入额: {result['行业总融资买入额']:,.2f}")
        print(f"行业内股票数量: {result['行业内股票数量']}")
        print(f"行业平均融资买入额: {result['行业平均融资买入额']:,.2f}")
        print(f"融资干扰因子: {result['融资干扰因子']:.4f}")
        
        # 保存结果到文件
        results_dir = ".\\results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存为CSV
        pd.DataFrame([result]).to_csv(f"{results_dir}\\{symbol}_融资干扰因子.csv", index=False, encoding="utf-8-sig")
        
        print(f"\n结果已保存到 {results_dir}\\{symbol}_融资干扰因子.csv")

