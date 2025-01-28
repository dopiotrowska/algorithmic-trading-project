import backtrader as bt
import yfinance as yf

class TrendFollowing(bt.Strategy):
    def __init__(self, stop_loss, take_profit, trailing_stop, max_positions):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.max_positions = max_positions
        self.open_positions = []
        self.high_prices = {}
        self.macd = bt.indicators.MACD(self.data.close)
        self.rsi = bt.indicators.RSI(self.data.close)
        self.atr = bt.indicators.ATR(self.data)
        self.atr_rolling_mean = bt.indicators.SimpleMovingAverage(self.atr, period=30)

    def next(self):
        if len(self.data) < 30:
            return

        print(f"Number of open positions: {len(self.open_positions)}")

        if len(self.open_positions) < self.max_positions:
            if (self.macd.macd[0] > self.macd.signal[0] and 
                self.rsi[0] > 50 and 
                self.atr[0] > self.atr_rolling_mean[0]):
                self.buy()
                self.open_positions.append({
                    'entry_price': self.data.close[0],
                    'stop_loss': self.data.close[0] * (1 - self.stop_loss),
                    'take_profit': self.data.close[0] * (1 + self.take_profit),
                })

        for pos in self.open_positions[:]:
            current_price = self.data.close[0]

            if current_price <= pos['stop_loss']:
                print(f"Stop-loss triggered at {current_price:.2f}")
                self.close()
                self.open_positions.remove(pos)
            elif current_price >= pos['take_profit']:
                print(f"Take-profit triggered at {current_price:.2f}")
                self.close()
                self.open_positions.remove(pos)
            else:
                self.high_prices[pos['entry_price']] = max(self.high_prices.get(pos['entry_price'], current_price), current_price)
                trailing_stop_price = self.high_prices[pos['entry_price']] * (1 - self.trailing_stop)
                if current_price < trailing_stop_price:
                    print(f"Trailing stop triggered at {current_price:.2f}")
                    self.close()
                    self.open_positions.remove(pos)

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                print(f"BUY EXECUTED: {order.executed.size} @ {order.executed.price}")
            elif order.issell():
                print(f"SELL EXECUTED: {order.executed.size} @ {order.executed.price}")

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"TRADE CLOSED: Gross PnL: {trade.pnl}, Net PnL: {trade.pnlcomm}")

def run_backtest(symbol, start_date, end_date, interval, stop_loss, take_profit, trailing_stop, max_positions=10, initial_capital=100000, slippage=0.002, commission=0.004, percents=10):
    cerebro = bt.Cerebro()
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for {symbol} in the specified date range.")
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.set_slippage_perc(slippage)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=percents)
    cerebro.addstrategy(TrendFollowing, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop, max_positions=max_positions)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    sharpe_ratio = results[0].analyzers.sharperatio.get_analysis().get('sharperatio', None)
    drawdown = results[0].analyzers.drawdown.get_analysis()
    return_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: None")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    cerebro.plot()

run_backtest(
    symbol="AAPL",
    start_date="2025-01-19",
    end_date="2025-01-27",
    interval="1m",
    stop_loss=0.02,
    take_profit=0.05,
    trailing_stop=0.03,
    max_positions=10,
    initial_capital=100000,
    slippage=0.002,
    commission=0.004,
    percents=10
)
