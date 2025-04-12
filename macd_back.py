from datetime import datetime, timedelta

import backtrader as bt  # 升级到最新版
import matplotlib.pyplot as plt  # 由于 Backtrader 的问题，此处要求 pip install matplotlib==3.2.2
import akshare as ak  # 升级到最新版
import pandas as pd

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

class MACDCrossStrategy(bt.Strategy):
    """
    MACD金叉死叉策略：仅使用MACD金叉作为买入信号，死叉作为卖出信号
    """
    params = (
        ("macd_fast", 12),      # MACD快线周期
        ("macd_slow", 26),      # MACD慢线周期
        ("macd_signal", 9),     # MACD信号线周期
    )

    def __init__(self):
        # MACD指标
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        
        # 订单相关
        self.order = None
        self.buy_price = None
        
        # 记录交易状态
        self.trade_count = 0
        self.profitable_trades = 0
        
        # 记录账户信息
        self.initial_cash = self.broker.getcash()
        
        # 添加T+1交易限制
        self.buy_date = None  # 记录买入日期
        self.can_sell = False  # 是否可以卖出
        
        # 交易记录
        self.trade_history = []

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
                self.buy_price = order.executed.price
                self.log(f'买入执行 [MACD金叉], 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
            else:
                sell_price = order.executed.price
                sell_size = order.executed.size
                profit_loss = (sell_price - self.buy_price) * sell_size
                profit_loss_pct = (sell_price / self.buy_price - 1) * 100 if self.buy_price else 0
                
                self.log(f'卖出执行 [MACD死叉], 价格: {sell_price:.2f}, 数量: {sell_size}, 收入: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
                self.log(f'交易结果 - 盈亏: {profit_loss:.2f} ({profit_loss_pct:.2f}%)')
                
                # 记录交易历史
                self.trade_history.append({
                    'buy_date': self.buy_date,
                    'sell_date': self.datetime.date(0),
                    'buy_price': self.buy_price,
                    'sell_price': sell_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct
                })
                
                if sell_price > self.buy_price:
                    self.profitable_trades += 1
                self.trade_count += 1
                
                # 打印账户信息
                self.log(f'账户信息 - 现金: {self.broker.getcash():.2f}, 持仓市值: {self.broker.getvalue() - self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
            
        self.order = None

    def check_macd_signal(self):
        """检查MACD金叉和死叉信号"""
        # 确保有足够的数据点
        if len(self.macd.macd) < 2 or len(self.macd.signal) < 2:
            return False, False
        
        # MACD金叉 - 快线从下向上穿过慢线
        golden_cross = (self.macd.macd[0] > self.macd.signal[0] and 
                        self.macd.macd[-1] <= self.macd.signal[-1])
        
        # MACD死叉 - 快线从上向下穿过慢线
        death_cross = (self.macd.macd[0] < self.macd.signal[0] and 
                       self.macd.macd[-1] >= self.macd.signal[-1])
        
        # 打印MACD状态便于调试
        if golden_cross or death_cross:
            close_price = self.datas[0].close[0]
            if golden_cross:
                self.log(f"检测到MACD金叉: MACD={self.macd.macd[0]:.4f}, Signal={self.macd.signal[0]:.4f}, 收盘价={close_price:.2f}")
            if death_cross:
                self.log(f"检测到MACD死叉: MACD={self.macd.macd[0]:.4f}, Signal={self.macd.signal[0]:.4f}, 收盘价={close_price:.2f}")
        
        return golden_cross, death_cross

    def next(self):
        """策略逻辑"""
        # 如果有未完成的订单，直接返回
        if self.order:
            return
            
        # 更新T+1状态
        current_date = self.datetime.date(0)
        if self.buy_date is not None:
            self.can_sell = current_date > self.buy_date
        
        # 检查MACD信号
        golden_cross, death_cross = self.check_macd_signal()
        
        # 如果已经持有股票
        if self.position:
            # 只有满足T+1才能卖出
            if not self.can_sell:
                return
            
            # 如果有死叉信号，执行卖出
            if death_cross:
                self.log(f'MACD死叉信号')
                self.order = self.sell(size=self.position.size)  # 全仓卖出
                self.buy_date = None
                self.can_sell = False
        
        # 如果没有持仓
        else:
            # 如果有金叉信号，执行买入
            if golden_cross:
                # 计算可买入的100股整数倍
                available_cash = self.broker.getcash()
                current_price = self.datas[0].close[0]
                max_shares = int((available_cash * 0.99) / current_price)  # 预留1%手续费
                shares = (max_shares // 100) * 100  # 向下取整到100的整数倍
                
                if shares >= 100:  # 如果能买至少100股
                    self.log(f'MACD金叉信号, 价格: {current_price:.2f}')
                    self.order = self.buy(size=shares)
                    self.buy_date = self.datetime.date(0)
                    self.can_sell = False
    
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
                self.log(f"交易 {i}: 买入日期 {trade['buy_date']}, 卖出日期 {trade['sell_date']}, "
                        f"买入价 {trade['buy_price']:.2f}, 卖出价 {trade['sell_price']:.2f}, "
                        f"盈亏 {trade['profit_loss']:.2f} ({trade['profit_loss_pct']:.2f}%)")

def run_backtest(stock_code, start_date, end_date, initial_cash=10000):
    """运行回测"""
    cerebro = bt.Cerebro()
    
    # 获取股票数据
    data = get_stock_data(stock_code, start_date, end_date)
    
    if data is None or data.empty:
        print(f"获取股票 {stock_code} 数据失败，无法进行回测")
        return None
    
    # 添加数据到回测系统
    feed = bt.feeds.PandasData(
        dataname=data,
        fromdate=start_date,
        todate=end_date
    )
    cerebro.adddata(feed)
    
    # 添加策略
    cerebro.addstrategy(MACDCrossStrategy)
    
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
    
    # 绘制结果
    cerebro.plot(style='candle', barup='red', bardown='green', volume=True, grid=True)
    
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
    # 参数设置
    stock_code = '601952'  # 股票代码（苏垦农发）
    initial_cash = 10000
    start_date = datetime(2024, 9, 1)
    end_date = datetime(2025, 2, 27)
    
    print(f"开始对股票 {stock_code} 进行MACD策略回测")
    
    # 运行策略
    result = run_backtest(stock_code, start_date, end_date, initial_cash)
    
    if result:
        print(f"\n===== 策略回测总结 =====")
        print(f"股票代码: {stock_code}")
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