import backtrader as bt
import yfinance as yf

class TrendFollowing(bt.Strategy):
    params = dict(
        stop_loss=0.02,         # 2% stop-loss
        take_profit=0.05,       # 5% take-profit
        trailing_stop=0.03,     # 3% trailing stop
        max_positions=10        # Maximum number of positions
    )

    def __init__(self):
        self.open_positions = []  # Track open positions
        self.high_prices = {}     # Track highest prices for trailing stops

        # Indicators
        self.macd = bt.indicators.MACD(self.data.close)
        self.rsi = bt.indicators.RSI(self.data.close)
        self.atr = bt.indicators.ATR(self.data)
        self.atr_rolling_mean = bt.indicators.SimpleMovingAverage(self.atr, period=30)

    def next(self):
        # Ensure enough data is available
        if len(self.data) < 30:
            return

        # Print the number of open positions
        print(f"Number of open positions: {len(self.open_positions)}")

        # --- Entry Logic ---
        if len(self.open_positions) < self.params.max_positions:
            # Buy signal: MACD bullish, RSI > 50, ATR > ATR mean
            if (self.macd.macd[0] > self.macd.signal[0] and 
                self.rsi[0] > 50 and 
                self.atr[0] > self.atr_rolling_mean[0]):
                self.buy()  # Automatically allocates 10% of available capital

                # Track this new position
                self.open_positions.append({
                    'entry_price': self.data.close[0],
                    'stop_loss': self.data.close[0] * (1 - self.params.stop_loss),
                    'take_profit': self.data.close[0] * (1 + self.params.take_profit),
                })

        # --- Exit Logic ---
        for pos in self.open_positions[:]:  # Use a copy to modify safely
            current_price = self.data.close[0]

            # Stop-loss condition
            if current_price <= pos['stop_loss']:
                print(f"Stop-loss triggered at {current_price:.2f}")
                self.close()  # Close the position
                self.open_positions.remove(pos)

            # Take-profit condition
            elif current_price >= pos['take_profit']:
                print(f"Take-profit triggered at {current_price:.2f}")
                self.close()  # Close the position
                self.open_positions.remove(pos)

            # Trailing stop logic
            else:
                self.high_prices[pos['entry_price']] = max(self.high_prices.get(pos['entry_price'], current_price), current_price)
                trailing_stop_price = self.high_prices[pos['entry_price']] * (1 - self.params.trailing_stop)
                if current_price < trailing_stop_price:
                    print(f"Trailing stop triggered at {current_price:.2f}")
                    self.close()  # Close the position
                    self.open_positions.remove(pos)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: {order.executed.size} @ {order.executed.price}")
            elif order.issell():
                print(f"SELL EXECUTED: {order.executed.size} @ {order.executed.price}")

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"TRADE CLOSED: Gross PnL: {trade.pnl}, Net PnL: {trade.pnlcomm}")

# Backtesting Function
def run_backtest(symbol, start_date, end_date, interval, stop_loss, take_profit, trailing_stop, initial_capital=100000, slippage=0.002, commission=0.004, percents=10):
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Fetch data from Yahoo Finance using yfinance
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for {symbol} in the specified date range.")
    data_feed = bt.feeds.PandasData(dataname=data)

    # Add the data to Cerebro
    cerebro.adddata(data_feed)

    # Set broker settings
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.set_slippage_perc(slippage)
    cerebro.broker.setcommission(commission=commission)

    # Set position size using PercentSizer
    cerebro.addsizer(bt.sizers.PercentSizer, percents=percents)

    # Add the strategy
    cerebro.addstrategy(TrendFollowing, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run the backtest
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()

    # Extract performance metrics
    sharpe_ratio = results[0].analyzers.sharperatio.get_analysis().get('sharperatio', None)
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # Print performance metrics
    return_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    if sharpe_ratio is not None:
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Sharpe Ratio: None")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print()

    # Plot the results
    cerebro.plot()

# Example Usage
run_backtest(
    symbol="AAPL",
    start_date="2025-01-19",
    end_date="2025-01-27",
    interval="1m",
    stop_loss=0.02,  # 2% stop loss
    take_profit=0.05,  # 5% take profit
    trailing_stop=0.03,  # 3% trailing stop
    initial_capital=100000,
    slippage=0.002,  # 0.2% slippage
    commission=0.004,  # 0.4% commission
    percents=10  # Automatically allocate 10% capital per trade
)
