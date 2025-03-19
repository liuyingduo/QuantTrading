import akshare as ak

stock_individual_basic_info_xq_df = ak.stock_individual_basic_info_xq(symbol="SH601127",token="4070a625ad532428a076ebb053b345d0e3fda53a")
#根据item列和value列生成字典
stock_individual_basic_info_xq_dict = dict(zip(stock_individual_basic_info_xq_df['item'],stock_individual_basic_info_xq_df['value']))
print(stock_individual_basic_info_xq_dict["main_operation_business"])
