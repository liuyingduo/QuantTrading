import akshare as ak
import akshare as ak

stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
# 按照涨跌幅排序
stock_board_industry_name_em_df = stock_board_industry_name_em_df.sort_values(by="涨跌幅", ascending=False)

print(stock_board_industry_name_em_df)
# 保存数据
stock_board_industry_name_em_df.to_excel("行业板块涨跌幅排名.xlsx", index=False)
