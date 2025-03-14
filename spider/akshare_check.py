import akshare as ak

stock_600519 = ak.stock_zh_a_hist(
    symbol="600519",#需要获得数据的股票代码
    period="daily",#数据周期，一般有分/日/周/月
    start_date="20200101",#数据起始日期
    end_date='20231201',#数据结束日期
    adjust="hfq")#数据格式

print(stock_600519)