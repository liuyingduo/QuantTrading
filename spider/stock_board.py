import akshare as ak
import json

stock_board_list = []
stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
# 循环获取每个版块的成分股
for index, row in stock_board_industry_name_em_df.iterrows():
    industry_name = row["板块名称"]
    print(f"正在获取 {industry_name} 板块的成分股数据...")
    
    # 获取成分股数据
    stock_board_industry_cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
    stock_board_industry_cons_df["板块名称"] = industry_name  # 添加板块名称列
    stock_board_industry_cons_df["板块代码"] = row["板块代码"]  # 添加行业名称列
    
    # 将成分股数据转变为列表字典 存储到json文件
    stock_board_list.extend(stock_board_industry_cons_df.to_dict(orient="records"))
    print(f"{industry_name} 板块数据获取完成！")

# 保存数据到JSON文件
with open(f".\\data\\stock_board.json", "w", encoding="utf-8") as f:
        json.dump(stock_board_list, f, ensure_ascii=False, indent=2)
    