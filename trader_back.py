from datetime import datetime, timedelta

import backtrader as bt  # 升级到最新版
import matplotlib.pyplot as plt  # 由于 Backtrader 的问题，此处要求 pip install matplotlib==3.2.2
import akshare as ak  # 升级到最新版
import pandas as pd
import concurrent.futures
from functools import partial

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 设置 K 线图颜色，红涨绿跌
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal'])

def get_stock_data(stock_code, start_date, end_date):
    """获取股票数据"""
    try:
        # 利用 AKShare 获取股票的后复权数据
        stock_hfq_df = ak.stock_zh_a_hist(symbol=stock_code, adjust="qfq").iloc[:, :7]
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
        return stock_hfq_df
    except Exception as e:
        print(f"获取股票{stock_code}数据失败：{str(e)}")
        return None

def get_low_price_stocks():
    """获取所有10元以下且以60或000开头的股票代码"""
    try:
        # 获取所有A股的实时行情
        stock_df = ak.stock_zh_a_spot_em()
        
        # 筛选出以60或000开头的股票
        condition = (stock_df['代码'].str.startswith('60') | stock_df['代码'].str.startswith('000'))
        filtered_stocks = stock_df[condition]
        
        # 再筛选出10元以下的股票
        low_price_stocks = filtered_stocks[filtered_stocks['最新价'] < 10]['代码'].tolist()
        
        print(f"找到{len(low_price_stocks)}只低于10元的60开头和000开头股票")
        return low_price_stocks
    except Exception as e:
        print(f"获取低价股票列表失败：{str(e)}")
        # 返回一些示例股票代码
        return ['600300', '000001', '000002']

def get_stock_pool():
    """获取更大范围的股票池，支持自定义条件"""
    try:
        # 获取所有A股的实时行情
        stock_df = ak.stock_zh_a_spot_em()
        
        # 默认返回所有股票代码
        all_stocks = stock_df['代码'].tolist()
        print(f"总共找到{len(all_stocks)}只股票")
        
        # 按板块分类统计
        sh_main = stock_df[stock_df['代码'].str.startswith('60')]['代码'].tolist()  # 上海主板
        sz_main = stock_df[stock_df['代码'].str.startswith('000')]['代码'].tolist()  # 深圳主板
        sz_gem = stock_df[stock_df['代码'].str.startswith('300')]['代码'].tolist()   # 创业板
        sh_star = stock_df[stock_df['代码'].str.startswith('688')]['代码'].tolist()  # 科创板
        
        print(f"上海主板(60开头): {len(sh_main)}只")
        print(f"深圳主板(000开头): {len(sz_main)}只")
        print(f"创业板(300开头): {len(sz_gem)}只")
        print(f"科创板(688开头): {len(sh_star)}只")
        
        return all_stocks
    except Exception as e:
        print(f"获取股票池失败：{str(e)}")
        return []

class HighReturnStrategy(bt.Strategy):
    """
    高收益策略：专注于捕捉高点卖出和低点买入，追求最大收益率
    """
    params = (
        ("sma5", 5),
        ("sma10", 10),
        ("sma20", 20),
        ("sma60", 60),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("atr_period", 14),
        ("atr_multiplier", 3.0),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 60),
        ("volume_ratio", 1.5),
        ("profit_take", 0.20),
        ("min_holding_days", 2),
        ("max_holding_days", 15),
        ("trailing_stop", True),
        ("trailing_percent", 0.05),
        ("profit_lock_threshold", 0.15),
        ("profit_lock_trailing", 0.02),
        ("ma_convergence_threshold", 0.01),  # 均线聚合阈值，均线之间的距离小于此值视为聚合
        ("ma_convergence_profit_take", 0.25),  # 均线聚合策略的止盈目标
        ("ma_convergence_max_holding_days", 25),  # 均线聚合策略的最大持仓天数
        ("trade_start_date", datetime(2025, 1, 1)),  # 添加交易开始时间参数
        ("holding_days", 3),  # 持有天数
    )

    def __init__(self):
        # 存储每个数据源（每支股票）的指标
        self.indicators = {}
        self.trend = {}
        
        # 遍历所有数据源（每支股票）
        for i, d in enumerate(self.datas):
            # 存储每个股票的指标
            self.indicators[d] = {
                # 价格数据
                'close': d.close,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'volume': d.volume,
                
                # 均线指标
                'sma5': bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma5),
                'sma10': bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma10),
                'sma20': bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma20),
                'sma60': bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma60),
                
                # 均线斜率
                'sma5_slope': bt.indicators.ROC(bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma5), period=1),
                'sma10_slope': bt.indicators.ROC(bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma10), period=1),
                'sma20_slope': bt.indicators.ROC(bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma20), period=1),
                
                # MACD指标
                'macd': bt.indicators.MACD(
                    d.close,
                    period_me1=self.params.macd_fast,
                    period_me2=self.params.macd_slow,
                    period_signal=self.params.macd_signal
                ),
                
                # MACD柱状图
                'macd_hist': bt.indicators.MACDHisto(
                    d.close,
                    period_me1=self.params.macd_fast,
                    period_me2=self.params.macd_slow,
                    period_signal=self.params.macd_signal
                ),
                
                # RSI指标
                'rsi': bt.indicators.RSI(d.close, period=self.params.rsi_period),
                
                # ATR指标
                'atr': bt.indicators.ATR(d, period=self.params.atr_period),
                
                # 成交量均线
                'volume_ma': bt.indicators.SimpleMovingAverage(d.volume, period=20),
                
                # 布林带指标
                'bollinger': bt.indicators.BollingerBands(d.close, period=20, devfactor=2.0),
                
                # 价格动量
                'momentum': bt.indicators.Momentum(d.close, period=5),
            }
            
            # 初始化趋势状态
            self.trend[d] = 'unknown'
        
        # 订单相关
        self.order = None
        self.buy_price = None
        self.stop_loss = None
        self.profit_target = None
        self.buy_signal_type = None  # 记录买入信号类型
        self.holding_days_count = 0  # 持仓天数计数
        self.highest_price = 0       # 持仓期间最高价格（用于追踪止损）
        self.profit_locked = False   # 是否已锁定部分利润
        
        # 记录交易状态
        self.trade_count = 0
        self.profitable_trades = 0
        self.last_sell_date = None
        self.cooldown_period = 0     # 修改为0，取消卖出后的冷却期
        
        # 记录账户信息
        self.initial_cash = self.broker.getcash()
        self.current_position = 0
        
        # 当前交易的股票
        self.current_stock = None
        
        # 添加T+1交易限制
        self.buy_date = None  # 记录买入日期
        self.can_sell = False  # 是否可以卖出
        
        # 添加交易开始日期检查
        self.can_trade = False  # 是否可以开始交易
        
        # 交易记录
        self.trade_history = []
        
        # 强制买入标志
        self.must_buy = False  # 新增：卖出后强制在下一个交易日买入

    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.current_position = order.executed.size
                stock_name = order.data._name
                self.log(f'买入执行 [{stock_name}] [{self.buy_signal_type}], 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                # 设置ATR止损
                self.stop_loss = self.buy_price - self.params.atr_multiplier * self.indicators[order.data]['atr'][0]
                # 设置止盈目标
                if "均线聚合" in self.buy_signal_type:
                    self.profit_target = self.buy_price * (1 + self.params.ma_convergence_profit_take)
                    self.log(f'均线聚合策略 - 设置更高止盈目标: {self.profit_target:.2f} (收益率: {self.params.ma_convergence_profit_take*100:.1f}%)')
                else:
                    self.profit_target = self.buy_price * (1 + self.params.profit_take)
                # 重置持仓天数和最高价
                self.holding_days_count = 0
                self.highest_price = self.buy_price
                self.profit_locked = False
                # 取消强制买入标志
                self.must_buy = False
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
            else:
                sell_price = order.executed.price
                sell_size = order.executed.size
                stock_name = order.data._name
                profit_loss = (sell_price - self.buy_price) * sell_size
                profit_loss_pct = (sell_price / self.buy_price - 1) * 100 if self.buy_price else 0
                
                self.log(f'卖出执行 [{stock_name}], 价格: {sell_price:.2f}, 数量: {sell_size}, 收入: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.log(f'交易结果 - 盈亏: {profit_loss:.2f} ({profit_loss_pct:.2f}%), 持仓时间: {self.holding_days_count}天, 最高价比例: {(self.highest_price/self.buy_price-1)*100:.2f}%')
                
                # 记录交易历史
                self.trade_history.append({
                    'stock': stock_name,
                    'buy_date': self.buy_date,
                    'sell_date': self.datetime.date(0),
                    'buy_price': self.buy_price,
                    'sell_price': sell_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'holding_days': self.holding_days_count
                })
                
                if sell_price > self.buy_price:
                    self.profitable_trades += 1
                self.trade_count += 1
                self.last_sell_date = self.datetime.date(0)
                self.current_position = 0
                self.current_stock = None
                
                # 激活强制买入标志，确保下一个交易日买入
                self.must_buy = True
                self.log(f'已激活强制买入标志，将在下一交易日买入最佳股票')
                
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None

    def update_trend(self, data):
        """更新当前市场趋势状态"""
        ind = self.indicators[data]
        # 判断中短期趋势
        if ind['sma5'][0] > ind['sma10'][0] > ind['sma20'][0]:
            if ind['sma5_slope'][0] > 0 and ind['sma10_slope'][0] > 0:
                self.trend[data] = 'up'
            else:
                self.trend[data] = 'sideways'
        elif ind['sma5'][0] < ind['sma10'][0] < ind['sma20'][0]:
            if ind['sma5_slope'][0] < 0 and ind['sma10_slope'][0] < 0:
                self.trend[data] = 'down'
            else:
                self.trend[data] = 'sideways'
        else:
            self.trend[data] = 'sideways'
            
    def detect_potential_top(self, data, prev_close, prev_high, consecutive_lower_closes, consecutive_lower_highs):
        """检测潜在的顶部形态"""
        ind = self.indicators[data]
        
        # 更新连续下跌计数
        if len(ind['close']) > 1 and prev_close > 0:
            if ind['close'][0] < prev_close:
                consecutive_lower_closes += 1
            else:
                consecutive_lower_closes = 0
        
        # 更新连续更低高点计数
        if len(ind['high']) > 1 and prev_high > 0:
            if ind['high'][0] < prev_high:
                consecutive_lower_highs += 1
            else:
                consecutive_lower_highs = 0
        
        # 1. 价格接近布林带上轨
        near_upper_band = ind['close'][0] > ind['bollinger'].lines.top[0] * 0.95
        
        # 2. RSI进入超买区域
        rsi_overbought = ind['rsi'][0] > self.params.rsi_overbought
        
        # 3. 价格动量减弱
        momentum_weakening = False
        if len(ind['momentum']) > 1:
            momentum_weakening = ind['momentum'][0] < ind['momentum'][-1]
        
        # 4. MACD柱状图减弱
        macd_weakening = False
        if len(ind['macd_hist']) > 1:
            macd_weakening = ind['macd_hist'][0] > 0 and ind['macd_hist'][0] < ind['macd_hist'][-1]
        
        # 5. 价格与短期均线距离过大（过度扩张）
        price_overextended = ind['close'][0] > ind['sma5'][0] * 1.05
        
        # 综合判断
        potential_top = (
            (near_upper_band and rsi_overbought) or
            (rsi_overbought and macd_weakening) or
            (consecutive_lower_closes >= 2 and rsi_overbought) or
            (consecutive_lower_highs >= 2 and momentum_weakening) or
            (price_overextended and momentum_weakening)
        )
        
        return potential_top, consecutive_lower_closes, consecutive_lower_highs, ind['close'][0], ind['high'][0]

    def check_buy_signal(self, data):
        """检查买入信号"""
        ind = self.indicators[data]
        
        # 更新趋势状态
        self.update_trend(data)
        
        # 检查股票价格是否在10元以下
        if ind['close'][0] >= 10:
            return False, None, 0
            
        # === 买入条件 ===
        
        # 1. 强势突破买入 - 适合上升趋势
        # 三线黄金排列 + 均线同向上涨 + 价格站上所有均线 + (MACD金叉或成交量确认)
        golden_cross = ind['sma5'][0] > ind['sma10'][0] > ind['sma20'][0]
        all_ma_rising = (ind['sma5_slope'][0] > 0 and ind['sma10_slope'][0] > 0 and ind['sma20_slope'][0] > 0)
        price_above_all_ma = (ind['close'][0] > ind['sma5'][0] and ind['close'][0] > ind['sma10'][0])
        
        # 确保有足够的数据点
        macd_golden_cross = False
        if len(ind['macd'].macd) > 1 and len(ind['macd'].signal) > 1:
            macd_golden_cross = (ind['macd'].macd[0] > ind['macd'].signal[0] and 
                                ind['macd'].macd[-1] <= ind['macd'].signal[-1])
        
        volume_confirm = ind['volume'][0] > ind['volume_ma'][0] * self.params.volume_ratio
        
        breakout_signal = (golden_cross and 
                           all_ma_rising and 
                           price_above_all_ma and 
                           (macd_golden_cross or volume_confirm))
        
        # 2. 低吸买入 - 适合回调买入
        # RSI超卖 + 价格接近布林带下轨 + 价格企稳
        rsi_oversold = ind['rsi'][0] < self.params.rsi_oversold
        near_lower_band = ind['close'][0] < ind['bollinger'].lines.bot[0] * 1.05
        
        price_stabilizing = False
        if len(ind['close']) > 1 and len(ind['open']) > 1:
            price_stabilizing = (ind['close'][0] > ind['open'][0] and 
                                ind['close'][-1] > ind['open'][-1])
        
        # 价格反弹确认 - 连续两天收阳
        price_bounce = False
        if len(ind['close']) > 2 and len(ind['open']) > 2:
            price_bounce = (ind['close'][0] > ind['open'][0] and 
                           ind['close'][-1] > ind['open'][-1] and
                           ind['close'][0] > ind['close'][-1])
        
        dip_buy_signal = (
            (rsi_oversold or near_lower_band) and 
            (price_stabilizing or price_bounce) and
            self.trend[data] != 'down'  # 不在下跌趋势中
        )
        
        # 3. 均线聚合买入 - 适合三条均线接近且趋势向上的情况
        # 计算均线之间的距离
        ma5_ma10_distance = abs(ind['sma5'][0] - ind['sma10'][0]) / ind['sma5'][0]
        ma10_ma20_distance = abs(ind['sma10'][0] - ind['sma20'][0]) / ind['sma10'][0]
        
        # 均线聚合条件：三条均线之间的距离都小于设定阈值
        ma_convergence = ma5_ma10_distance < self.params.ma_convergence_threshold and ma10_ma20_distance < self.params.ma_convergence_threshold
        
        # 价格站上所有均线
        price_above_mas = ind['close'][0] > ind['sma5'][0] and ind['close'][0] > ind['sma10'][0] and ind['close'][0] > ind['sma20'][0]
        
        # 均线方向向上
        ma_direction_up = ind['sma5_slope'][0] > 0 and ind['sma10_slope'][0] > 0
        
        # MACD柱状图为正或向上
        macd_positive_or_rising = False
        if len(ind['macd_hist']) > 1:
            macd_positive_or_rising = ind['macd_hist'][0] > 0 or (ind['macd_hist'][0] > ind['macd_hist'][-1])
        
        # 均线聚合买入信号
        ma_convergence_signal = (
            ma_convergence and 
            price_above_mas and 
            ma_direction_up and 
            macd_positive_or_rising and
            self.trend[data] != 'down'
        )
        
        # 如果满足均线聚合条件，记录日志
        if ma_convergence:
            self.log(f'[{data._name}] 均线聚合检测 - 距离: MA5-MA10={ma5_ma10_distance:.4f}, MA10-MA20={ma10_ma20_distance:.4f}, 阈值={self.params.ma_convergence_threshold:.4f}')
            if ma_convergence_signal:
                self.log(f'[{data._name}] 均线聚合买入信号 - 价格站上均线: {price_above_mas}, 均线向上: {ma_direction_up}, MACD状态: {macd_positive_or_rising}, 趋势: {self.trend[data]}')
        
        # 必须有均线聚合信号，其他信号作为辅助确认
        buy_signal = ma_convergence_signal and (breakout_signal or dip_buy_signal)
        
        # 计算得分，用于在多只股票都有买入信号时选择最佳的
        score = 0
        signal_type = ""
        
        if buy_signal:
            # 基础分数 - 由均线聚合和辅助信号提供
            score += 50  # 均线聚合基础分
            
            # 根据均线聚合的紧密度加分
            convergence_score = (1.0 - (ma5_ma10_distance + ma10_ma20_distance) / (2 * self.params.ma_convergence_threshold)) * 20
            score += max(0, convergence_score)
            
            # 根据趋势加分
            if self.trend[data] == 'up':
                score += 15
            elif self.trend[data] == 'sideways':
                score += 5
                
            # 根据价格位置加分
            if price_above_mas:
                score += 10
                
            # 根据成交量加分
            if volume_confirm:
                score += 10
                
            # 根据MACD加分
            if macd_golden_cross:
                score += 10
            elif macd_positive_or_rising:
                score += 5
                
            # 根据RSI加分
            if 40 <= ind['rsi'][0] <= 60:  # 理想的RSI范围
                score += 10
            elif 30 <= ind['rsi'][0] < 40 or 60 < ind['rsi'][0] <= 70:  # 可接受的RSI范围
                score += 5
                
            # 确定信号类型
            if breakout_signal:
                signal_type = "均线聚合+强势突破"
            elif dip_buy_signal:
                signal_type = "均线聚合+低吸回调"
        
        return buy_signal, signal_type, score

    def next(self):
        """策略逻辑"""
        if self.order:
            return
            
        # 检查是否达到交易开始时间
        current_date = self.datetime.date(0)
        if not self.can_trade and current_date >= self.params.trade_start_date.date():
            self.can_trade = True
            self.log(f"开始交易，当前日期：{current_date}")
            # 启动时激活强制买入标志
            self.must_buy = True
            
        # 如果未达到交易开始时间，直接返回
        if not self.can_trade:
            return
            
        # 更新T+1状态
        if self.buy_date is not None:
            self.can_sell = current_date > self.buy_date
        
        # 持仓状态 - 更新持仓天数和检查卖出条件
        if self.position:
            # 更新持仓天数和最高价
            self.holding_days_count += 1
            current_price = self.indicators[self.current_stock]['close'][0]
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # 计算当前持仓盈亏百分比
            current_profit_pct = (current_price / self.buy_price - 1) * 100
            
            # 更新追踪止损价格
            if self.params.trailing_stop and current_price > self.buy_price:
                # 根据利润水平选择不同的追踪止损比例
                if current_profit_pct >= self.params.profit_lock_threshold and not self.profit_locked:
                    # 当利润达到阈值时，启用更激进的追踪止损
                    self.profit_locked = True
                    self.log(f'利润锁定机制激活 - 当前利润: {current_profit_pct:.2f}%, 追踪止损比例: {self.params.profit_lock_trailing * 100}%')
                
                # 选择追踪止损比例
                trailing_percent = self.params.profit_lock_trailing if self.profit_locked else self.params.trailing_percent
                
                # 计算新的追踪止损价格
                trailing_stop_price = self.highest_price * (1 - trailing_percent)
                
                # 如果新的追踪止损价格高于当前止损价格，则更新止损价格
                if trailing_stop_price > self.stop_loss:
                    self.stop_loss = trailing_stop_price
            
            # 检查是否满足T+1
            if not self.can_sell:
                return
                
            # === 卖出条件 ===
            
            # 1. 止损：价格低于止损线
            stop_loss_triggered = current_price < self.stop_loss
            
            # 2. 固定持仓天数后卖出
            fixed_holding_days_reached = self.holding_days_count >= self.params.holding_days
            
            # 卖出信号：止损或达到持仓天数
            sell_signal = stop_loss_triggered or fixed_holding_days_reached
            
            if sell_signal:
                sell_reason = ""
                if stop_loss_triggered:
                    sell_reason = "止损"
                elif fixed_holding_days_reached:
                    sell_reason = f"持仓{self.params.holding_days}天后卖出"
                
                self.log(f'卖出信号触发 [{self.current_stock._name}] - 原因: {sell_reason}, 当前价: {current_price:.2f}, 买入价: {self.buy_price:.2f}, 最高价: {self.highest_price:.2f}, 盈亏: {current_profit_pct:.2f}%')
                self.order = self.sell(self.current_stock, size=self.current_position)  # 全仓卖出
                self.buy_date = None
                self.can_sell = False
                
        # 不持仓状态 - 寻找买入机会
        elif not self.position:
            # 检查是否在卖出冷却期内 - 只有在非强制买入模式下才检查
            if not self.must_buy and self.last_sell_date is not None:
                days_since_sell = (self.datetime.date(0) - self.last_sell_date).days
                if days_since_sell < self.cooldown_period:
                    return
            
            # 用于存储符合买入条件的股票及其得分
            buy_candidates = []
            
            # 遍历所有数据，检查买入信号
            for data in self.datas:
                # 检查买入信号
                prev_close = prev_high = 0
                consecutive_lower_closes = consecutive_lower_highs = 0
                
                # 检测潜在顶部，避免买在高点
                potential_top, consecutive_lower_closes, consecutive_lower_highs, prev_close, prev_high = \
                    self.detect_potential_top(data, prev_close, prev_high, consecutive_lower_closes, consecutive_lower_highs)
                
                if potential_top:
                    continue  # 如果检测到潜在顶部，不买入该股票
                
                # 检查该股票是否满足买入条件
                buy_signal, signal_type, score = self.check_buy_signal(data)
                
                # 如果处于强制买入模式，放宽买入条件
                if self.must_buy:
                    # 放宽买入条件，只要均线聚合就可以考虑
                    ind = self.indicators[data]
                    ma5_ma10_distance = abs(ind['sma5'][0] - ind['sma10'][0]) / ind['sma5'][0]
                    ma10_ma20_distance = abs(ind['sma10'][0] - ind['sma20'][0]) / ind['sma10'][0]
                    
                    if (ma5_ma10_distance < self.params.ma_convergence_threshold * 1.5 and 
                        ma10_ma20_distance < self.params.ma_convergence_threshold * 1.5 and
                        ind['close'][0] < 10):  # 确保价格在10元以下
                        
                        # 计算一个基础得分
                        base_score = 50
                        if ind['close'][0] > ind['sma5'][0] and ind['close'][0] > ind['sma10'][0]:
                            base_score += 10  # 价格站上短期均线加分
                        
                        if ind['sma5_slope'][0] > 0:
                            base_score += 5  # 短期均线向上加分
                            
                        if self.trend[data] == 'up':
                            base_score += 5  # 上升趋势加分
                            
                        # 使用放宽的条件和基础得分
                        signal_type = "强制买入-均线接近"
                        buy_signal = True
                        score = base_score
                        
                        self.log(f'强制买入模式 - [{data._name}] 满足放宽条件，得分: {score}')
                
                if buy_signal:
                    # 计算可买入的100股整数倍
                    available_cash = self.broker.getcash()
                    price = self.indicators[data]['close'][0]
                    max_shares = int((available_cash * 0.99) / price)  # 预留1%手续费
                    shares = (max_shares // 100) * 100  # 向下取整到100的整数倍
                    
                    if shares >= 100:  # 如果能买至少100股
                        buy_candidates.append({
                            'data': data,
                            'score': score,
                            'signal_type': signal_type,
                            'price': price,
                            'shares': shares
                        })
            
            # 如果有符合条件的股票，选择得分最高的买入
            if buy_candidates:
                # 按得分排序
                buy_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_candidate = buy_candidates[0]
                
                self.log(f"选择最佳买入股票: {best_candidate['data']._name}, 得分: {best_candidate['score']}, 信号: {best_candidate['signal_type']}, 价格: {best_candidate['price']:.2f}")
                
                # 执行买入
                self.buy_signal_type = best_candidate['signal_type']
                self.order = self.buy(best_candidate['data'], size=best_candidate['shares'])
                self.buy_date = self.datetime.date(0)
                self.can_sell = False
                self.current_stock = best_candidate['data']
            elif self.must_buy:
                # 如果强制买入模式下没有找到符合条件的股票，输出日志
                self.log("强制买入模式下未找到符合条件的股票，将在下一交易日继续尝试")
                
    def stop(self):
        """策略结束时执行"""
        # 确保交易计数正确
        actual_trade_count = len(self.trade_history)
        win_rate = (self.profitable_trades / actual_trade_count * 100) if actual_trade_count > 0 else 0
        total_return = (self.broker.getvalue() / self.initial_cash - 1) * 100
        self.log(f'策略结束 - 总交易次数: {actual_trade_count}, 盈利交易: {self.profitable_trades}, 胜率: {win_rate:.2f}%, 总收益率: {total_return:.2f}%')
        
        # 打印交易历史
        if self.trade_history:
            self.log("===== 交易历史 =====")
            for i, trade in enumerate(self.trade_history, 1):
                self.log(f"交易 {i}: 股票 {trade['stock']}, 买入日期 {trade['buy_date']}, 卖出日期 {trade['sell_date']}, "
                        f"买入价 {trade['buy_price']:.2f}, 卖出价 {trade['sell_price']:.2f}, "
                        f"盈亏 {trade['profit_loss']:.2f} ({trade['profit_loss_pct']:.2f}%), 持仓 {trade['holding_days']}天")

def run_strategy(start_date, end_date, initial_cash=10000):
    """运行回测"""
    cerebro = bt.Cerebro()
    
    # 获取所有低价股票
    stock_list = get_low_price_stocks()
    
    # 如果股票列表为空或过少，尝试获取更多股票
    if len(stock_list) < 10:
        print("警告：符合条件的股票太少，尝试获取更多股票...")
        
        # 获取更多股票作为备选
        all_stocks = get_stock_pool()
        
        # 简单随机抽样一些股票以确保有足够的标的
        import random
        if len(all_stocks) > 0:
            random_stocks = random.sample(all_stocks, min(50, len(all_stocks)))
            stock_list.extend([s for s in random_stocks if s not in stock_list])
            print(f"扩展股票池至{len(stock_list)}只股票")
    
    # 限制股票数量，以提高回测速度
    # max_stocks = 50  # 可以根据实际情况调整
    # if len(stock_list) > max_stocks:
    #     import random
    #     stock_list = random.sample(stock_list, max_stocks)
    #     print(f"为提高回测速度，随机选择{max_stocks}只股票进行测试")
    
    # 数据获取开始时间
    data_start_date = datetime(2024, 1, 1)
    
    # 并行获取所有股票数据
    stock_data_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {executor.submit(get_stock_data, stock_code, data_start_date, end_date): stock_code 
                          for stock_code in stock_list}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_code = future_to_stock[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    stock_data_dict[stock_code] = data
            except Exception as e:
                print(f"处理股票{stock_code}时出错：{str(e)}")
    
    # 添加所有股票数据到cerebro
    for stock_code, data in stock_data_dict.items():
        feed = bt.feeds.PandasData(
            dataname=data,
            fromdate=data_start_date,
            todate=end_date,
            name=stock_code  # 设置数据源的名称为股票代码
        )
        cerebro.adddata(feed)
    
    # 添加策略
    cerebro.addstrategy(HighReturnStrategy, trade_start_date=start_date, holding_days=3)
    
    # 设置初始资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.002)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 获取回测结果
    port_value = cerebro.broker.getvalue()
    pnl = port_value - initial_cash
    
    return {
        'final_value': port_value,
        'pnl': pnl,
        'sharpe': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
        'annual_return': strat.analyzers.returns.get_analysis().get('rnorm100', 0),
        'trade_count': len(strat.trade_history),
        'win_rate': (strat.profitable_trades / len(strat.trade_history) * 100) if strat.trade_history else 0
    }

def main():
    initial_cash = 10000
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 13)
    
    # 运行策略
    result = run_strategy(start_date, end_date, initial_cash)
    
    if result:
        print(f"\n===== 策略回测总结 =====")
        print(f"初始资金: {initial_cash}")
        print(f"最终资金: {round(result['final_value'], 2)}")
        print(f"总收益: {round(result['pnl'], 2)}")
        print(f"总收益率: {round(result['pnl'] / initial_cash * 100, 2)}%")
        print(f"交易笔数: {result['trade_count']}")
        print(f"胜率: {round(result['win_rate'], 2)}%")
        print(f"夏普比率: {result['sharpe'] if result['sharpe'] is None else round(result['sharpe'], 2)}")
        print(f"最大回撤: {round(result['max_drawdown'], 2)}%")
        print(f"年化收益率: {round(result['annual_return'], 2)}%")

if __name__ == "__main__":
    main()