from datetime import datetime

import backtrader as bt  # 升级到最新版
import matplotlib.pyplot as plt  # 由于 Backtrader 的问题，此处要求 pip install matplotlib==3.2.2
import akshare as ak  # 升级到最新版
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 设置 K 线图颜色，红涨绿跌
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal'])

# 利用 AKShare 获取股票的后复权数据，这里只获取前 7 列
stock_hfq_df = ak.stock_zh_a_hist(symbol="600163", adjust="hfq").iloc[:, :7]
# 删除 `股票代码` 列
del stock_hfq_df['股票代码']
# 处理字段命名，以符合 Backtrader 的要求
stock_hfq_df.columns = [
    'date',
    'open',
    'close',
    'high',
    'low',
    'volume',
]
# 把 date 作为日期索引，以符合 Backtrader 的要求
stock_hfq_df.index = pd.to_datetime(stock_hfq_df['date'])


class MyStrategy(bt.Strategy):
    """
    主策略程序
    """
    params = (
        ("sma5", 5),
        ("sma10", 10),
        ("sma20", 20),
    )  # 全局设定交易策略的参数

    def __init__(self):
        """
        初始化函数
        """
        self.data_close = self.datas[0].close  # 指定价格序列
        self.data_open = self.datas[0].open    # 指定开盘价序列
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
        # 添加移动均线指标
        self.sma5 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma5
        )
        self.sma10 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma10
        )
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma20
        )
        
        # 创建交叉信号指标
        self.crossover5_10 = bt.indicators.CrossOver(self.sma5, self.sma10)  # 5日均线上穿10日均线
        self.crossover10_20 = bt.indicators.CrossOver(self.sma10, self.sma20)  # 10日均线上穿/下穿20日均线

    def next(self):
        """
        执行逻辑
        """
        if self.order:  # 检查是否有指令等待执行
            return
            
        # 检查是否持仓
        if not self.position:  # 没有持仓
            # 执行买入条件判断：5日均线上穿10日均线
            if self.crossover5_10 > 0:
                self.order = self.buy(size=100)  # 执行买入
        else:
            # 执行卖出条件判断：10日均线跌破20日均线
            if self.crossover10_20 < 0:
                self.order = self.sell(size=100)  # 执行卖出


class AdvancedStrategy(bt.Strategy):
    """
    高级策略：结合MACD、RSI和ATR的多重信号策略
    """
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_period", 14),
        ("rsi_overbought", 70),
        ("rsi_oversold", 30),
        ("atr_period", 14),
        ("atr_multiplier", 2.0),  # ATR乘数，用于计算止损
        ("volume_period", 20),    # 成交量均线周期
    )

    def __init__(self):
        # 价格数据
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume
        
        # 订单相关
        self.order = None
        self.buy_price = None
        self.stop_loss = None
        
        # MACD指标
        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        # 计算MACD柱状图
        self.macd_hist = bt.indicators.MACDHisto(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        # 创建MACD柱状图的交叉信号
        self.macd_cross = bt.indicators.CrossOver(self.macd_hist, 0)
        
        # RSI指标
        self.rsi = bt.indicators.RSI(
            self.data_close, 
            period=self.params.rsi_period
        )
        
        # ATR指标 - 用于计算波动率和设置止损
        self.atr = bt.indicators.ATR(
            self.datas[0], 
            period=self.params.atr_period
        )
        
        # 成交量均线
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data_volume, 
            period=self.params.volume_period
        )
        
        # 布林带指标
        self.bollinger = bt.indicators.BollingerBands(
            self.data_close, 
            period=20, 
            devfactor=2.0
        )
        
        # 记录交易状态
        self.in_trade = False
        self.trade_count = 0
        self.profitable_trades = 0

    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                # 设置ATR止损
                self.stop_loss = self.buy_price - self.params.atr_multiplier * self.atr[0]
                self.in_trade = True
            else:
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                if order.executed.price > self.buy_price:
                    self.profitable_trades += 1
                self.trade_count += 1
                self.in_trade = False
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None

    def next(self):
        """策略逻辑"""
        if self.order:
            return
            
        # 检查是否持仓
        if not self.position:
            # === 买入条件 ===
            # 1. MACD柱状图由负转正
            macd_cross_up = self.macd_cross > 0
            
            # 2. RSI从超卖区域回升
            rsi_cross_up = self.rsi[0] > self.params.rsi_oversold and self.rsi[-1] <= self.params.rsi_oversold
            
            # 3. 价格突破布林带上轨
            price_break_upper = self.data_close[0] > self.bollinger.lines.top[0] and self.data_close[-1] <= self.bollinger.lines.top[-1]
            
            # 4. 成交量放大
            volume_increase = self.data_volume[0] > 1.5 * self.volume_ma[0]
            
            # 综合买入信号
            if (macd_cross_up or rsi_cross_up or price_break_upper) and volume_increase:
                self.log(f'买入信号触发 - MACD: {macd_cross_up}, RSI: {rsi_cross_up}, 布林带: {price_break_upper}, 成交量: {volume_increase}')
                self.order = self.buy(size=100)
        else:
            # === 卖出条件 ===
            # 1. 止损：价格低于ATR止损线
            if self.data_close[0] < self.stop_loss:
                self.log(f'触发止损卖出, 当前价: {self.data_close[0]:.2f}, 止损价: {self.stop_loss:.2f}')
                self.order = self.sell(size=100)
                return
                
            # 2. MACD柱状图由正转负
            macd_cross_down = self.macd_cross < 0
            
            # 3. RSI进入超买区域后回落
            rsi_cross_down = self.rsi[0] < self.params.rsi_overbought and self.rsi[-1] >= self.params.rsi_overbought
            
            # 4. 价格跌破布林带中轨
            price_break_middle = self.data_close[0] < self.bollinger.lines.mid[0] and self.data_close[-1] >= self.bollinger.lines.mid[-1]
            
            # 综合卖出信号
            if macd_cross_down or rsi_cross_down or price_break_middle:
                self.log(f'卖出信号触发 - MACD: {macd_cross_down}, RSI: {rsi_cross_down}, 布林带: {price_break_middle}')
                self.order = self.sell(size=100)
                
    def stop(self):
        """策略结束时执行"""
        win_rate = self.profitable_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        self.log(f'策略结束 - 总交易次数: {self.trade_count}, 盈利交易: {self.profitable_trades}, 胜率: {win_rate:.2f}%')


class BalancedStrategy(bt.Strategy):
    """
    平衡策略：结合均线趋势和MACD动量，简化信号条件，增加仓位管理
    增强低谷买入能力，特别优化对极端低点的捕捉
    """
    params = (
        ("sma20", 20),
        ("sma60", 60),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("atr_period", 14),
        ("atr_multiplier", 3.0),  # 增大ATR乘数，减少止损频率
        ("risk_ratio", 0.02),     # 风险比例，用于计算仓位
        ("rsi_period", 14),       # RSI周期
        ("rsi_oversold", 35),     # RSI超卖阈值（放宽到35）
        ("rsi_extreme_oversold", 25),  # RSI极度超卖阈值（放宽到25）
        ("bb_period", 20),        # 布林带周期
        ("bb_dev", 2.0),          # 布林带标准差倍数
        ("price_drop_pct", 5.0),  # 价格大幅下跌百分比
        ("volume_spike", 1.5),    # 成交量放大倍数
    )

    def __init__(self):
        # 价格数据
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume
        
        # 订单相关
        self.order = None
        self.buy_price = None
        self.stop_loss = None
        
        # 均线指标 - 用于判断趋势
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma20
        )
        self.sma60 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma60
        )
        
        # 短期均线 - 用于捕捉短期反弹
        self.sma5 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=5
        )
        self.sma10 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=10
        )
        
        # 均线交叉信号
        self.sma_cross = bt.indicators.CrossOver(self.sma20, self.sma60)
        self.sma_short_cross = bt.indicators.CrossOver(self.sma5, self.sma10)
        
        # MACD指标 - 用于确认动量
        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # MACD柱状图交叉信号
        self.macd_hist = bt.indicators.MACDHisto(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # MACD交叉信号
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        
        # RSI指标 - 用于捕捉超卖反弹
        self.rsi = bt.indicators.RSI(
            self.data_close, 
            period=self.params.rsi_period
        )
        
        # 布林带指标 - 用于识别价格极值
        self.bollinger = bt.indicators.BollingerBands(
            self.data_close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
        
        # 价格与布林带下轨的距离百分比
        self.bb_pct = bt.indicators.PercentRank(
            (self.data_close - self.bollinger.lines.bot) / self.data_close * 100,
            period=50
        )
        
        # ATR指标 - 用于计算波动率和设置止损
        self.atr = bt.indicators.ATR(
            self.datas[0], 
            period=self.params.atr_period
        )
        
        # 成交量均线
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data_volume, 
            period=20
        )
        
        # 记录交易状态
        self.trade_count = 0
        self.profitable_trades = 0
        self.last_sell_date = None
        self.cooldown_period = 2  # 进一步减少卖出后的冷却期（天数）
        
        # 低谷检测变量
        self.low_point_count = 0  # 连续低点计数
        self.prev_low = 0         # 前一个低点价格
        self.lowest_price = 999999  # 记录观察期内的最低价
        self.highest_price = 0      # 记录观察期内的最高价
        self.price_history = []     # 记录价格历史

    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                # 设置ATR止损
                self.stop_loss = self.buy_price - self.params.atr_multiplier * self.atr[0]
            else:
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                if order.executed.price > self.buy_price:
                    self.profitable_trades += 1
                self.trade_count += 1
                self.last_sell_date = self.datas[0].datetime.date(0)
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None

    def next(self):
        """策略逻辑"""
        if self.order:
            return
            
        # 更新价格历史
        self.price_history.append(self.data_close[0])
        if len(self.price_history) > 60:  # 保持最近60天的价格历史
            self.price_history.pop(0)
            
        # 更新最高最低价
        if len(self.price_history) > 20:
            self.lowest_price = min(self.price_history[-20:])
            self.highest_price = max(self.price_history[-20:])
            
        # 检查是否持仓
        if not self.position:
            # 检查是否在卖出冷却期内
            if self.last_sell_date is not None:
                days_since_sell = (self.datas[0].datetime.date(0) - self.last_sell_date).days
                if days_since_sell < self.cooldown_period:
                    return
            
            # === 低谷检测逻辑 ===
            # 1. 检测连续下跌后的企稳
            price_stabilizing = self.data_close[0] > self.data_close[-1] and self.data_close[-1] < self.data_close[-2]
            
            # 2. 检测价格是否接近布林带下轨
            near_bb_bottom = self.data_close[0] < self.bollinger.lines.bot[0] * 1.05
            
            # 3. 检测极度超卖
            extreme_oversold = self.rsi[0] < self.params.rsi_extreme_oversold
            
            # 4. 检测价格创新低后反弹
            new_low_bounce = False
            if len(self.data_close) > 20:  # 确保有足够的历史数据
                recent_low = min([self.data_low[-i] for i in range(1, 11)])  # 最近10天的最低价
                if self.data_low[-1] <= recent_low and self.data_close[0] > self.data_open[0]:
                    new_low_bounce = True
            
            # 5. 检测大幅下跌后的反弹（新增）
            sharp_drop_bounce = False
            if len(self.price_history) > 5:
                five_day_high = max(self.price_history[-5:])
                drop_pct = (five_day_high - self.data_low[-1]) / five_day_high * 100
                if drop_pct > self.params.price_drop_pct and self.data_close[0] > self.data_open[0]:
                    sharp_drop_bounce = True
            
            # 6. 检测价格接近近期低点（新增）
            near_recent_low = False
            if len(self.price_history) > 20:
                if self.data_close[0] < self.lowest_price * 1.03:  # 价格在最低点的3%以内
                    near_recent_low = True
            
            # 7. 检测成交量放大（新增）
            volume_spike = self.data_volume[0] > self.volume_ma[0] * self.params.volume_spike
            
            # 8. 检测价格跌破所有均线（新增）
            price_below_all_mas = (self.data_close[0] < self.sma5[0] and 
                                  self.data_close[0] < self.sma10[0] and 
                                  self.data_close[0] < self.sma20[0])
            
            # === 买入条件 ===
            # 1. 中期趋势向上：20日均线在60日均线上方
            trend_up = self.sma20[0] > self.sma60[0]
            
            # 2. MACD柱状图为正值或刚转为正值
            macd_positive = self.macd_hist[0] > 0
            
            # 3. MACD金叉：MACD线上穿信号线
            macd_golden_cross = self.macd_cross > 0
            
            # 4. 短期均线金叉：5日均线上穿10日均线
            short_term_up = self.sma_short_cross > 0
            
            # 5. RSI超卖反弹
            rsi_bounce = self.rsi[0] > self.params.rsi_oversold and self.rsi[-1] <= self.params.rsi_oversold
            
            # 6. 价格连续上涨
            price_rising = self.data_close[0] > self.data_close[-1] > self.data_close[-2]
            
            # 7. 低谷反转信号：RSI极度超卖 + 价格企稳
            bottom_reversal = ((extreme_oversold and price_stabilizing) or 
                              (near_bb_bottom and price_stabilizing) or 
                              new_low_bounce or 
                              sharp_drop_bounce or  # 新增
                              (near_recent_low and price_stabilizing))  # 新增
            
            # 8. 恐慌抛售信号（新增）：价格大幅下跌 + 成交量放大 + 价格跌破所有均线
            panic_sell = sharp_drop_bounce and volume_spike and price_below_all_mas
            
            # 综合买入信号：满足以下条件之一
            # A. 中期趋势向上 + MACD确认
            # B. MACD金叉 + 短期均线金叉
            # C. RSI超卖反弹 + 价格连续上涨
            # D. 低谷反转信号
            # E. 恐慌抛售信号（新增）
            buy_signal = (trend_up and macd_positive) or \
                         (macd_golden_cross and short_term_up) or \
                         (rsi_bounce and price_rising) or \
                         bottom_reversal or \
                         panic_sell
            
            # 特殊情况：价格处于极低位置时，放宽买入条件（新增）
            if near_recent_low and self.rsi[0] < 40:
                buy_signal = True
            
            if buy_signal:
                # 计算仓位大小：基于ATR的风险管理
                risk_per_share = self.atr[0] * self.params.atr_multiplier
                if risk_per_share > 0:
                    # 在低谷买入时增加仓位
                    risk_multiplier = 2.0 if panic_sell else 1.5 if bottom_reversal else 1.0
                    risk_amount = self.broker.getvalue() * self.params.risk_ratio * risk_multiplier
                    size = max(1, int(risk_amount / risk_per_share))
                    
                    signal_type = ""
                    if trend_up and macd_positive:
                        signal_type = "趋势信号"
                    elif macd_golden_cross and short_term_up:
                        signal_type = "短期反弹信号"
                    elif rsi_bounce and price_rising:
                        signal_type = "RSI反弹信号"
                    elif bottom_reversal:
                        signal_type = "低谷反转信号"
                    elif panic_sell:
                        signal_type = "恐慌抛售信号"
                    elif near_recent_low and self.rsi[0] < 40:
                        signal_type = "极低价格信号"
                    
                    self.log(f'买入信号触发 - {signal_type}, 仓位: {size}')
                    self.order = self.buy(size=size)
                else:
                    self.order = self.buy(size=100)  # 默认仓位
        else:
            # === 卖出条件 ===
            # 1. 止损：价格低于ATR止损线
            if self.data_close[0] < self.stop_loss:
                self.log(f'触发止损卖出, 当前价: {self.data_close[0]:.2f}, 止损价: {self.stop_loss:.2f}')
                self.order = self.sell()
                return
                
            # 2. 趋势反转：20日均线跌破60日均线
            if self.sma_cross < 0:
                self.log(f'趋势反转卖出, 20日均线跌破60日均线')
                self.order = self.sell()
                return
                
            # 3. MACD柱状图由正转负
            if self.macd_hist[0] < 0 and self.macd_hist[-1] > 0:
                self.log(f'MACD反转卖出, MACD柱状图由正转负')
                self.order = self.sell()
                return
                
            # 4. MACD死叉：MACD线下穿信号线（在MACD为正值区域）
            if self.macd_cross < 0 and self.macd.macd[0] > 0:
                self.log(f'MACD死叉卖出, MACD线下穿信号线')
                self.order = self.sell()
                
    def stop(self):
        """策略结束时执行"""
        win_rate = self.profitable_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        self.log(f'策略结束 - 总交易次数: {self.trade_count}, 盈利交易: {self.profitable_trades}, 胜率: {win_rate:.2f}%')


cerebro = bt.Cerebro()  # 初始化回测系统
start_date = datetime(2023, 11, 3)  # 回测开始时间
end_date = datetime(2025, 3, 13)  # 回测结束时间
data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
cerebro.adddata(data)  # 将数据传入回测系统

# 选择使用哪个策略
# cerebro.addstrategy(MyStrategy)  # 简单均线交叉策略
# cerebro.addstrategy(AdvancedStrategy)  # 高级多指标策略
cerebro.addstrategy(BalancedStrategy)  # 平衡策略

start_cash = 1000000
cerebro.broker.setcash(start_cash)  # 设置初始资本为 100000
cerebro.broker.setcommission(commission=0.002)  # 设置交易手续费为 0.2%

# 添加分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')  # 夏普比率
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')   # 最大回撤
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')     # 收益率

# 添加观察者
cerebro.addobserver(bt.observers.Broker)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.DrawDown)

# 运行回测
results = cerebro.run()  # 运行回测系统
strat = results[0]

# 获取回测结果
port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
pnl = port_value - start_cash  # 盈亏统计
sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
annual_return = strat.analyzers.returns.get_analysis()['rnorm100']

print(f"初始资金: {start_cash}\n回测期间：{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}")
print(f"总资金: {round(port_value, 2)}")
print(f"净收益: {round(pnl, 2)}")
print(f"夏普比率: {round(sharpe, 2)}")
print(f"最大回撤: {round(max_dd, 2)}%")
print(f"年化收益率: {round(annual_return, 2)}%")

# 设置红涨绿跌
cerebro.plot(style='candlestick', barup='red', bardown='green', valuetags=False)