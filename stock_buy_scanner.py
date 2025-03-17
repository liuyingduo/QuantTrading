import datetime
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import backtrader as bt
import os
import time
from concurrent.futures import ThreadPoolExecutor

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class BuySignalAnalyzer:
    """
    买入信号分析器：分析股票是否触发买入信号
    基于HighReturnStrategy的买入策略
    """
    def __init__(self, stock_code, stock_name=None, data_days=120):
        """
        初始化分析器
        
        参数:
        stock_code: 股票代码
        stock_name: 股票名称（可选）
        data_days: 获取的历史数据天数
        """
        self.stock_code = stock_code
        self.stock_name = stock_name if stock_name else stock_code
        self.data_days = data_days
        self.data = None
        self.buy_signals = []
        
        # 策略参数
        self.params = {
            "sma5": 5,
            "sma10": 10,
            "sma20": 20,
            "sma60": 60,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "volume_ratio": 1.5,
            "ma_convergence_threshold": 0.01,
        }
        
    def load_data(self):
        """加载股票数据"""
        try:
            # 使用AKShare获取股票数据
            stock_df = ak.stock_zh_a_hist(symbol=self.stock_code, adjust="qfq", period="daily")
            
            # 只保留需要的列
            stock_df = stock_df[['日期', '开盘', '收盘', '最高', '最低', '成交量']]
            stock_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            
            # 转换日期列为日期类型
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            
            # 只保留最近N天的数据
            if len(stock_df) > self.data_days:
                stock_df = stock_df.tail(self.data_days)
                
            # 设置日期为索引
            stock_df.set_index('date', inplace=True)
            
            self.data = stock_df
            return True
        except Exception as e:
            print(f"加载股票 {self.stock_code} 数据失败: {e}")
            return False
    
    def calculate_indicators(self):
        """计算技术指标"""
        if self.data is None or len(self.data) < 60:
            print(f"股票 {self.stock_code} 数据不足，无法计算指标")
            return False
            
        # 计算移动平均线
        self.data['sma5'] = self.data['close'].rolling(window=self.params['sma5']).mean()
        self.data['sma10'] = self.data['close'].rolling(window=self.params['sma10']).mean()
        self.data['sma20'] = self.data['close'].rolling(window=self.params['sma20']).mean()
        self.data['sma60'] = self.data['close'].rolling(window=self.params['sma60']).mean()
        
        # 计算均线斜率
        self.data['sma5_slope'] = self.data['sma5'].pct_change(1)
        self.data['sma10_slope'] = self.data['sma10'].pct_change(1)
        self.data['sma20_slope'] = self.data['sma20'].pct_change(1)
        
        # 计算MACD
        exp1 = self.data['close'].ewm(span=self.params['macd_fast'], adjust=False).mean()
        exp2 = self.data['close'].ewm(span=self.params['macd_slow'], adjust=False).mean()
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=self.params['macd_signal'], adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
        
        # 计算RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + 2 * std
        self.data['bb_lower'] = self.data['bb_middle'] - 2 * std
        
        # 计算成交量均线
        self.data['volume_ma20'] = self.data['volume'].rolling(window=20).mean()
        
        return True
    
    def analyze_buy_signals(self):
        """分析买入信号"""
        if self.data is None or 'sma5' not in self.data.columns:
            print(f"股票 {self.stock_code} 指标未计算，无法分析买入信号")
            return False
            
        # 获取最新的数据点
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else None
        
        # 初始化信号字典
        signals = {
            "强势突破": False,
            "低吸回调": False,
            "均线聚合": False,
            "信号详情": {}
        }
        
        # 1. 强势突破买入信号
        golden_cross = latest['sma5'] > latest['sma10'] > latest['sma20']
        all_ma_rising = (latest['sma5_slope'] > 0 and latest['sma10_slope'] > 0 and latest['sma20_slope'] > 0)
        price_above_all_ma = (latest['close'] > latest['sma5'] and latest['close'] > latest['sma10'])
        
        # MACD金叉
        macd_golden_cross = False
        if prev is not None:
            macd_golden_cross = (latest['macd'] > latest['macd_signal'] and 
                                prev['macd'] <= prev['macd_signal'])
        
        # 成交量确认
        volume_confirm = latest['volume'] > latest['volume_ma20'] * self.params['volume_ratio']
        
        # 强势突破信号
        breakout_signal = (golden_cross and 
                          all_ma_rising and 
                          price_above_all_ma and 
                          (macd_golden_cross or volume_confirm))
        
        signals["强势突破"] = breakout_signal
        signals["信号详情"]["强势突破"] = {
            "三线黄金排列": golden_cross,
            "均线同向上涨": all_ma_rising,
            "价格站上均线": price_above_all_ma,
            "MACD金叉": macd_golden_cross,
            "成交量确认": volume_confirm
        }
        
        # 2. 低吸回调买入信号
        rsi_oversold = latest['rsi'] < self.params['rsi_oversold']
        near_lower_band = latest['close'] < latest['bb_lower'] * 1.05
        
        # 价格企稳
        price_stabilizing = False
        if prev is not None:
            price_stabilizing = (latest['close'] > latest['open'] and 
                                prev['close'] > prev['open'])
        
        # 价格反弹确认
        price_bounce = False
        if len(self.data) > 2:
            prev2 = self.data.iloc[-3]
            price_bounce = (latest['close'] > latest['open'] and 
                           prev['close'] > prev['open'] and
                           latest['close'] > prev['close'])
        
        # 判断趋势
        trend = 'unknown'
        if latest['sma5'] > latest['sma10'] > latest['sma20']:
            if latest['sma5_slope'] > 0 and latest['sma10_slope'] > 0:
                trend = 'up'
            else:
                trend = 'sideways'
        elif latest['sma5'] < latest['sma10'] < latest['sma20']:
            if latest['sma5_slope'] < 0 and latest['sma10_slope'] < 0:
                trend = 'down'
            else:
                trend = 'sideways'
        else:
            trend = 'sideways'
        
        # 低吸回调信号
        dip_buy_signal = (
            (rsi_oversold or near_lower_band) and 
            (price_stabilizing or price_bounce) and
            trend != 'down'  # 不在下跌趋势中
        )
        
        signals["低吸回调"] = dip_buy_signal
        signals["信号详情"]["低吸回调"] = {
            "RSI超卖": rsi_oversold,
            "接近布林带下轨": near_lower_band,
            "价格企稳": price_stabilizing,
            "价格反弹": price_bounce,
            "非下跌趋势": trend != 'down'
        }
        
        # 3. 均线聚合买入信号
        # 计算均线之间的距离
        ma5_ma10_distance = abs(latest['sma5'] - latest['sma10']) / latest['sma5']
        ma10_ma20_distance = abs(latest['sma10'] - latest['sma20']) / latest['sma10']
        
        # 均线聚合条件
        ma_convergence = (ma5_ma10_distance < self.params['ma_convergence_threshold'] and 
                         ma10_ma20_distance < self.params['ma_convergence_threshold'])
        
        # 价格站上所有均线
        price_above_mas = (latest['close'] > latest['sma5'] and 
                          latest['close'] > latest['sma10'] and 
                          latest['close'] > latest['sma20'])
        
        # 均线方向向上
        ma_direction_up = latest['sma5_slope'] > 0 and latest['sma10_slope'] > 0
        
        # MACD柱状图为正或向上
        macd_positive_or_rising = False
        if prev is not None:
            macd_positive_or_rising = latest['macd_hist'] > 0 or (latest['macd_hist'] > prev['macd_hist'])
        
        # 均线聚合买入信号
        ma_convergence_signal = (
            ma_convergence and 
            price_above_mas and 
            ma_direction_up and 
            macd_positive_or_rising and
            trend != 'down'
        )
        
        signals["均线聚合"] = ma_convergence_signal
        signals["信号详情"]["均线聚合"] = {
            "均线聚合": ma_convergence,
            "MA5-MA10距离": ma5_ma10_distance,
            "MA10-MA20距离": ma10_ma20_distance,
            "价格站上均线": price_above_mas,
            "均线向上": ma_direction_up,
            "MACD状态": macd_positive_or_rising,
            "非下跌趋势": trend != 'down'
        }
        
        # 综合买入信号
        signals["任一买入信号"] = breakout_signal or dip_buy_signal or ma_convergence_signal
        
        self.buy_signals = signals
        return True
    
    def get_result(self):
        """获取分析结果"""
        if not self.buy_signals:
            return {
                "股票代码": self.stock_code,
                "股票名称": self.stock_name,
                "任一买入信号": False,
                "强势突破": False,
                "低吸回调": False,
                "均线聚合": False,
                "最新价格": None,
                "信号详情": {}
            }
        
        latest_price = self.data['close'].iloc[-1] if self.data is not None and len(self.data) > 0 else None
        
        return {
            "股票代码": self.stock_code,
            "股票名称": self.stock_name,
            "任一买入信号": self.buy_signals["任一买入信号"],
            "强势突破": self.buy_signals["强势突破"],
            "低吸回调": self.buy_signals["低吸回调"],
            "均线聚合": self.buy_signals["均线聚合"],
            "最新价格": latest_price,
            "信号详情": self.buy_signals["信号详情"]
        }
    
    def run_analysis(self):
        """运行完整分析流程"""
        if not self.load_data():
            return self.get_result()
        
        if not self.calculate_indicators():
            return self.get_result()
        
        self.analyze_buy_signals()
        return self.get_result()


class StockScanner:
    """
    股票扫描器：扫描股票列表，分析哪些股票触发了买入信号
    """
    def __init__(self, stock_list=None, use_index=False, index_code="000300", top_n=300):
        """
        初始化扫描器
        
        参数:
        stock_list: 股票列表，格式为 [(code, name), ...]
        use_index: 是否使用指数成分股
        index_code: 指数代码，默认为沪深300
        top_n: 使用指数成分股时，选取的股票数量
        """
        self.stock_list = stock_list if stock_list else []
        self.use_index = use_index
        self.index_code = index_code
        self.top_n = top_n
        self.results = []
        
    def load_stock_list(self):
        """加载股票列表"""
        if self.stock_list:
            return
            
        if self.use_index:
            try:
                # 获取指数成分股
                if self.index_code == "000300":
                    # 沪深300成分股
                    index_stocks = ak.index_stock_cons_weight_csindex(symbol="000300")
                    # 只保留股票代码和名称
                    index_stocks = index_stocks[['成分券代码', '成分券名称']].values.tolist()
                    # 只保留前N个
                    self.stock_list = index_stocks[:self.top_n]
                else:
                    # 其他指数可以根据需要添加
                    print(f"暂不支持指数 {self.index_code} 的成分股获取")
                    self.stock_list = []
            except Exception as e:
                print(f"获取指数成分股失败: {e}")
                self.stock_list = []
        else:
            try:
                # 获取A股股票列表
                stock_list = ak.stock_zh_a_spot_em()
                # 只保留股票代码和名称
                stock_list = stock_list[['代码', '名称']].values.tolist()
                # 只保留前N个
                self.stock_list = stock_list[:self.top_n]
            except Exception as e:
                print(f"获取A股股票列表失败: {e}")
                self.stock_list = []
    
    def analyze_single_stock(self, stock):
        """分析单个股票"""
        code, name = stock
        analyzer = BuySignalAnalyzer(code, name)
        result = analyzer.run_analysis()
        return result
    
    def scan_stocks(self, max_workers=10):
        """扫描所有股票"""
        self.load_stock_list()
        
        if not self.stock_list:
            print("股票列表为空，无法进行扫描")
            return []
        
        print(f"开始扫描 {len(self.stock_list)} 只股票...")
        start_time = time.time()
        
        # 使用多线程加速处理
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.analyze_single_stock, stock) for stock in self.stock_list]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 打印进度
                    if (i + 1) % 10 == 0 or (i + 1) == len(self.stock_list):
                        print(f"已处理 {i + 1}/{len(self.stock_list)} 只股票")
                except Exception as e:
                    print(f"处理股票 {self.stock_list[i]} 时出错: {e}")
        
        end_time = time.time()
        print(f"扫描完成，耗时 {end_time - start_time:.2f} 秒")
        
        # 按照是否有买入信号排序
        results.sort(key=lambda x: (not x["任一买入信号"], not x["强势突破"], not x["低吸回调"], not x["均线聚合"]))
        
        self.results = results
        return results
    
    def save_results(self, filename=None):
        """保存结果到CSV文件"""
        if not self.results:
            print("没有结果可保存")
            return
            
        if filename is None:
            # 使用当前日期作为文件名
            today = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"股票买入信号_{today}.csv"
        
        # 提取需要保存的字段
        data = []
        for result in self.results:
            data.append({
                "股票代码": result["股票代码"],
                "股票名称": result["股票名称"],
                "任一买入信号": result["任一买入信号"],
                "强势突破": result["强势突破"],
                "低吸回调": result["低吸回调"],
                "均线聚合": result["均线聚合"],
                "最新价格": result["最新价格"]
            })
        
        # 保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"结果已保存到 {filename}")
    
    def print_buy_signals(self):
        """打印有买入信号的股票"""
        if not self.results:
            print("没有分析结果")
            return
            
        buy_signals = [r for r in self.results if r["任一买入信号"]]
        
        if not buy_signals:
            print("没有股票触发买入信号")
            return
            
        print(f"\n共有 {len(buy_signals)} 只股票触发买入信号:")
        print("-" * 60)
        print(f"{'股票代码':<10}{'股票名称':<15}{'最新价格':<10}{'强势突破':<10}{'低吸回调':<10}{'均线聚合':<10}")
        print("-" * 60)
        
        for signal in buy_signals:
            print(f"{signal['股票代码']:<10}{signal['股票名称']:<15}{signal['最新价格']:<10.2f}"
                  f"{signal['强势突破']:<10}{signal['低吸回调']:<10}{signal['均线聚合']:<10}")
        
        print("-" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("股票买入信号扫描器")
    print("=" * 60)
    
    # 选择扫描模式
    print("\n请选择扫描模式:")
    print("1. 扫描沪深300成分股")
    print("2. 扫描自定义股票列表")
    print("3. 扫描单个股票")
    
    choice = input("请输入选择 (默认1): ").strip() or "1"
    
    if choice == "1":
        # 扫描沪深300成分股
        scanner = StockScanner(use_index=True, index_code="000300", top_n=300)
        scanner.scan_stocks()
        scanner.print_buy_signals()
        scanner.save_results()
    
    elif choice == "2":
        # 扫描自定义股票列表
        stock_file = input("请输入股票列表文件路径 (CSV格式，包含代码和名称列): ").strip()
        
        if not os.path.exists(stock_file):
            print(f"文件 {stock_file} 不存在")
            return
            
        try:
            # 读取CSV文件
            df = pd.read_csv(stock_file)
            # 获取股票代码和名称
            stock_list = df.iloc[:, :2].values.tolist()
            
            scanner = StockScanner(stock_list=stock_list)
            scanner.scan_stocks()
            scanner.print_buy_signals()
            scanner.save_results()
        except Exception as e:
            print(f"读取股票列表文件失败: {e}")
    
    elif choice == "3":
        # 扫描单个股票
        stock_code = input("请输入股票代码: ").strip()
        stock_name = input("请输入股票名称 (可选): ").strip() or stock_code
        
        analyzer = BuySignalAnalyzer(stock_code, stock_name)
        result = analyzer.run_analysis()
        
        print("\n分析结果:")
        print(f"股票代码: {result['股票代码']}")
        print(f"股票名称: {result['股票名称']}")
        print(f"最新价格: {result['最新价格']}")
        print(f"任一买入信号: {result['任一买入信号']}")
        print(f"强势突破: {result['强势突破']}")
        print(f"低吸回调: {result['低吸回调']}")
        print(f"均线聚合: {result['均线聚合']}")
        
        # 打印详细信号
        if result["任一买入信号"]:
            print("\n信号详情:")
            for signal_type, details in result["信号详情"].items():
                if result[signal_type]:
                    print(f"\n{signal_type}信号:")
                    for k, v in details.items():
                        print(f"  {k}: {v}")
    
    else:
        print("无效的选择")


if __name__ == "__main__":
    main() 