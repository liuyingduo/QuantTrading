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
    精英策略：专注于高胜率、高收益的交易机会
    采用精简的买入卖出条件，减少交易频率，提高单笔收益
    """
    params = (
        ("sma5", 5),        # 短期均线
        ("sma10", 10),      # 中期均线
        ("sma20", 20),      # 长期均线
        ("sma60", 60),      # 长期趋势均线
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("atr_period", 14),
        ("atr_multiplier", 3.0),  # ATR乘数，用于止损
        ("rsi_period", 14),       # RSI周期
        ("rsi_oversold", 30),     # RSI超卖阈值
        ("rsi_overbought", 70),   # RSI超买阈值
        ("volume_ratio", 1.5),    # 成交量放大比例
        ("profit_take", 0.20),    # 止盈比例（20%）
        ("min_holding_days", 5),  # 最小持仓天数
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
        self.profit_target = None
        self.buy_signal_type = None  # 记录买入信号类型
        self.holding_days = 0        # 持仓天数
        
        # 均线指标 - 三线系统
        self.sma5 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma5
        )
        self.sma10 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma10
        )
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma20
        )
        
        # 长期趋势均线 - 用于判断大趋势
        self.sma60 = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.params.sma60
        )
        
        # 均线斜率 - 用于判断趋势方向
        self.sma5_slope = bt.indicators.ROC(self.sma5, period=1)
        self.sma10_slope = bt.indicators.ROC(self.sma10, period=1)
        self.sma20_slope = bt.indicators.ROC(self.sma20, period=1)
        self.sma60_slope = bt.indicators.ROC(self.sma60, period=1)
        
        # MACD指标 - 用于确认动量
        self.macd = bt.indicators.MACD(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # MACD柱状图
        self.macd_hist = bt.indicators.MACDHisto(
            self.data_close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # RSI指标 - 用于判断超卖/超买
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
            period=20
        )
        
        # 布林带指标 - 用于判断价格波动范围
        self.bollinger = bt.indicators.BollingerBands(
            self.data_close, 
            period=20, 
            devfactor=2.0
        )
        
        # 记录交易状态
        self.trade_count = 0
        self.profitable_trades = 0
        self.last_sell_date = None
        self.cooldown_period = 3  # 增加卖出后的冷却期（天数）
        
        # 记录账户信息
        self.initial_cash = self.broker.getcash()
        self.current_position = 0
        
        # 趋势状态
        self.trend = None  # 'up', 'down', 'sideways'
        self.trend_changed_date = None
        
        # 市场状态评分
        self.market_score = 0  # 市场状态评分，用于判断买入时机

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
                self.current_position = order.executed.size
                self.log(f'买入执行 [{self.buy_signal_type}], 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                # 设置ATR止损
                self.stop_loss = self.buy_price - self.params.atr_multiplier * self.atr[0]
                # 设置止盈目标
                self.profit_target = self.buy_price * (1 + self.params.profit_take)
                # 重置持仓天数
                self.holding_days = 0
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
            else:
                sell_price = order.executed.price
                sell_size = order.executed.size
                profit_loss = (sell_price - self.buy_price) * sell_size
                profit_loss_pct = (sell_price / self.buy_price - 1) * 100 if self.buy_price else 0
                
                self.log(f'卖出执行, 价格: {sell_price:.2f}, 数量: {sell_size}, 收入: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.log(f'交易结果 - 盈亏: {profit_loss:.2f} ({profit_loss_pct:.2f}%), 持仓时间: {self.holding_days}天')
                
                if sell_price > self.buy_price:
                    self.profitable_trades += 1
                self.trade_count += 1
                self.last_sell_date = self.datas[0].datetime.date(0)
                self.current_position = 0
                
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None

    def update_trend(self):
        """更新当前市场趋势状态"""
        # 判断长期趋势
        long_term_trend_up = self.sma20[0] > self.sma60[0] and self.sma60_slope[0] > 0
        long_term_trend_down = self.sma20[0] < self.sma60[0] and self.sma60_slope[0] < 0
        
        # 判断中短期趋势
        short_term_trend_up = self.sma5[0] > self.sma10[0] and self.sma10[0] > self.sma20[0]
        short_term_trend_down = self.sma5[0] < self.sma10[0] and self.sma10[0] < self.sma20[0]
        
        # 判断均线斜率
        slopes_up = self.sma5_slope[0] > 0 and self.sma10_slope[0] > 0 and self.sma20_slope[0] > 0
        slopes_down = self.sma5_slope[0] < 0 and self.sma10_slope[0] < 0 and self.sma20_slope[0] < 0
        
        # 综合判断趋势
        prev_trend = self.trend
        
        if (long_term_trend_up and short_term_trend_up) or (short_term_trend_up and slopes_up):
            self.trend = 'up'
        elif (long_term_trend_down and short_term_trend_down) or (short_term_trend_down and slopes_down):
            self.trend = 'down'
        else:
            self.trend = 'sideways'
            
        # 记录趋势变化
        if prev_trend != self.trend:
            self.trend_changed_date = self.datas[0].datetime.date(0)
            self.log(f'趋势变化: {prev_trend} -> {self.trend}')
            
    def calculate_market_score(self):
        """计算市场状态评分，用于判断买入时机"""
        score = 0
        
        # 趋势评分 (0-40分)
        if self.trend == 'up':
            score += 40
        elif self.trend == 'sideways':
            score += 20
        
        # 均线排列评分 (0-20分)
        if self.sma5[0] > self.sma10[0] > self.sma20[0]:
            score += 20
        elif self.sma5[0] > self.sma10[0]:
            score += 10
            
        # MACD评分 (0-15分)
        if self.macd.macd[0] > self.macd.signal[0] and self.macd_hist[0] > 0:
            score += 15
        elif self.macd.macd[0] > self.macd.signal[0]:
            score += 10
        elif self.macd_hist[0] > 0:
            score += 5
            
        # RSI评分 (0-15分)
        if 40 <= self.rsi[0] <= 60:  # 中性区域，上升空间大
            score += 15
        elif 30 <= self.rsi[0] < 40:  # 轻度超卖
            score += 10
        elif self.rsi[0] > 60:  # 接近超买
            score += 5
            
        # 成交量评分 (0-10分)
        if self.data_volume[0] > self.volume_ma[0] * self.params.volume_ratio:
            score += 10
        elif self.data_volume[0] > self.volume_ma[0]:
            score += 5
            
        self.market_score = score
        return score

    def next(self):
        """策略逻辑"""
        if self.order:
            return
            
        # 更新趋势状态
        self.update_trend()
        
        # 计算市场评分
        self.calculate_market_score()
        
        # 检查是否持仓
        if not self.position:
            # 检查是否在卖出冷却期内
            if self.last_sell_date is not None:
                days_since_sell = (self.datas[0].datetime.date(0) - self.last_sell_date).days
                if days_since_sell < self.cooldown_period:
                    return
            
            # === 精英买入条件 ===
            
            # 1. 市场评分必须达到高分（至少70分）
            high_market_score = self.market_score >= 70
            
            # 2. 三线黄金排列 - 经典的强势上涨形态
            golden_cross = self.sma5[0] > self.sma10[0] > self.sma20[0] > self.sma60[0]
            
            # 3. 均线同向上涨 - 确认趋势强度
            all_ma_rising = (self.sma5_slope[0] > 0 and 
                            self.sma10_slope[0] > 0 and 
                            self.sma20_slope[0] > 0)
            
            # 4. 价格站上所有均线 - 确认价格强势
            price_above_all_ma = (self.data_close[0] > self.sma5[0] and 
                                 self.data_close[0] > self.sma10[0] and 
                                 self.data_close[0] > self.sma20[0])
            
            # 5. MACD金叉确认 - 动量指标确认
            macd_golden_cross = (self.macd.macd[0] > self.macd.signal[0] and 
                                self.macd.macd[-1] <= self.macd.signal[-1])
            
            # 6. 成交量确认 - 放量突破更可靠
            volume_confirm = self.data_volume[0] > self.volume_ma[0] * self.params.volume_ratio
            
            # 强势突破买入信号：市场评分高 + 三线黄金排列 + 均线同向上涨 + 价格站上所有均线 + (MACD金叉或成交量确认)
            breakout_signal = (high_market_score and 
                              golden_cross and 
                              all_ma_rising and 
                              price_above_all_ma and 
                              (macd_golden_cross or volume_confirm))
            
            # 买入信号
            buy_signal = breakout_signal
            
            if buy_signal:
                self.buy_signal_type = "强势突破"
                
                # 全仓买入
                cash = self.broker.getcash()
                price = self.data_close[0]
                size = int(cash * 0.99 / price)  # 预留1%现金应对手续费
                
                if size > 0:
                    self.log(f'买入信号触发 - 类型: {self.buy_signal_type}, 价格: {price:.2f}, 数量: {size}, 总金额: {price * size:.2f}, 市场评分: {self.market_score}')
                    self.order = self.buy(size=size)
                    self.last_buy_date = self.datas[0].datetime.date(0)
        else:
            # 更新持仓天数
            self.holding_days += 1
            
            # 计算当前持仓盈亏百分比
            current_profit_pct = (self.data_close[0] / self.buy_price - 1) * 100
            
            # === 精英卖出条件 ===
            
            # 1. 止损：价格低于ATR止损线（保留核心止损机制）
            stop_loss_triggered = self.data_close[0] < self.stop_loss
            
            # 2. 止盈：价格达到止盈目标（20%收益）
            take_profit_triggered = self.data_close[0] >= self.profit_target
            
            # 3. 趋势明确转为下跌
            trend_turned_down = (self.trend == 'down' and 
                                self.trend_changed_date == self.datas[0].datetime.date(0) and 
                                self.sma5[0] < self.sma10[0])
            
            # 4. 顶部反转信号：MACD死叉 + 价格跌破5日均线 + RSI从高位回落
            top_reversal = (self.macd.macd[0] < self.macd.signal[0] and 
                           self.macd.macd[-1] >= self.macd.signal[-1] and 
                           self.data_close[0] < self.sma5[0] and 
                           self.rsi[0] < self.rsi[-1] and self.rsi[-1] > 65)
            
            # 最小持仓天数检查
            min_holding_days_met = self.holding_days >= self.params.min_holding_days
            
            # 卖出信号：止损不受最小持仓天数限制，其他卖出条件需要满足最小持仓天数
            sell_signal = stop_loss_triggered or (min_holding_days_met and (take_profit_triggered or trend_turned_down or top_reversal))
            
            if sell_signal:
                sell_reason = ""
                if stop_loss_triggered:
                    sell_reason = "止损"
                elif take_profit_triggered:
                    sell_reason = "止盈"
                elif trend_turned_down:
                    sell_reason = "趋势转为下跌"
                elif top_reversal:
                    sell_reason = "顶部反转"
                
                self.log(f'卖出信号触发 - 原因: {sell_reason}, 当前价: {self.data_close[0]:.2f}, 买入价: {self.buy_price:.2f}, 盈亏: {current_profit_pct:.2f}%, 持仓天数: {self.holding_days}')
                self.order = self.sell(size=self.current_position)  # 全仓卖出
                
    def stop(self):
        """策略结束时执行"""
        win_rate = self.profitable_trades / self.trade_count * 100 if self.trade_count > 0 else 0
        total_return = (self.broker.getvalue() / self.initial_cash - 1) * 100
        self.log(f'策略结束 - 总交易次数: {self.trade_count}, 盈利交易: {self.profitable_trades}, 胜率: {win_rate:.2f}%, 总收益率: {total_return:.2f}%')


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