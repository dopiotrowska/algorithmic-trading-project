import backtrader as bt
import yfinance as yf
import datetime

# Create strategy class
class MovingAverageCrossover(bt.Strategy):
    def __init__(self, short_period, long_period, stop_loss, take_profit, trailing_stop):
        self.short_period = short_period
        self.long_period = long_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop

        # Moving Averages
        self.short_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.short_period)
        self.long_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.long_period)

        # To track the entry price of the trade
        self.entry_price = None

    def next(self):
        # Print current prices and indicators for debugging
        print(f"Short SMA: {self.short_sma[0]:.2f}, Long SMA: {self.long_sma[0]:.2f}")
        print(f"Close: {self.data.close[0]:.2f}")

        # Ensure enough data is available (check for long-period SMA initialization)
        if len(self.data) < self.long_period:
            return

        # Buy signal: short SMA crosses above long SMA
        if self.short_sma[0] > self.long_sma[0] and not self.position:
            print(f"Buy signal at {self.data.datetime.datetime()}")
            self.buy()  # Place the buy order
            self.entry_price = self.data.close[0]  # Record the entry price

        # Sell signal: short SMA crosses below long SMA
        elif self.short_sma[0] < self.long_sma[0] and self.position:
            print(f"Sell signal at {self.data.datetime.datetime()}")
            self.sell()  # Place the sell order

        # Implement stop loss
        if self.position and self.stop_loss:
            stop_loss_price = self.entry_price * (1 - self.stop_loss if self.position.size > 0 else 1 + self.stop_loss)
            if (self.position.size > 0 and self.data.close[0] < stop_loss_price) or (self.position.size < 0 and self.data.close[0] > stop_loss_price):
                print(f"Stop Loss triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()  # Close the position

        # Implement take profit
        if self.position and self.take_profit:
            take_profit_price = self.entry_price * (1 + self.take_profit if self.position.size > 0 else 1 - self.take_profit)
            if (self.position.size > 0 and self.data.close[0] > take_profit_price) or (self.position.size < 0 and self.data.close[0] < take_profit_price):
                print(f"Take Profit triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()  # Close the position

        # Implement trailing stop
        if self.position and self.trailing_stop:
            if self.position.size > 0:  # Long position
                # Track the highest price reached since entering the position
                highest_price = max(self.entry_price, self.data.close[0])
                trailing_stop_price = highest_price * (1 - self.trailing_stop)

                if self.data.close[0] < trailing_stop_price:
                    print(
                        f"Trailing Stop triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                    self.close()  # Close the position

            elif self.position.size < 0:  # Short position
                # Track the lowest price reached since entering the position
                lowest_price = min(self.entry_price, self.data.close[0])
                trailing_stop_price = lowest_price * (1 + self.trailing_stop)

                if self.data.close[0] > trailing_stop_price:
                    print(
                        f"Trailing Stop triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                    self.close()  # Close the position

# Method to run the backtest with custom parameters
def run_backtest(symbol, start_date, end_date, short_period, long_period, stop_loss, take_profit, trailing_stop, initial_capital=100000, slippage=0.001, commission=0.001, percents=10):
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Fetch data from Yahoo Finance using yfinance
    data = yf.download(symbol, start=start_date, end=end_date)

    # Save data to a CSV file
    csv_filename = f"{symbol}_data.csv"
    data.to_csv(csv_filename)

    # Load the CSV file into Backtrader
    data_feed = bt.feeds.PandasData(dataname=data)

    # Add the data to the Cerebro engine
    cerebro.adddata(data_feed)

    # Set initial capital
    cerebro.broker.set_cash(initial_capital)

    # Set commission and slippage
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)

    # Add position sizer (e.g., percents% of available capital per trade)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=percents)

    # Print the starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")

    # Add the strategy to the Cerebro engine, passing the parameters
    cerebro.addstrategy(MovingAverageCrossover, short_period=short_period, long_period=long_period, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)

    # Add analyzers for performance metrics (Sharpe Ratio and Drawdown)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="trans")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    initial_portfolio_value = cerebro.broker.getvalue()

    results = cerebro.run()

    # Extract analyzers results
    sharpe_ratio = results[0].analyzers.sharperatio.get_analysis()['sharperatio']
    drawdown = results[0].analyzers.drawdown.get_analysis()

    final_portfolio_value = cerebro.broker.getvalue()
    return_percentage = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100




    # Print the final portfolio value and performance metrics
    print(f"Initial Portfolio Value: {initial_portfolio_value}")
    print(f"Ending Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Max Drawdown Duration: {drawdown.max.len} days")

    # Plot the results
    cerebro.plot()

# Example usage of the method with custom parameters
symbol = "^GSPC" # ^GSPC, CDR.WA
start_date = "1960-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Call the backtest method with parameters
run_backtest(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    short_period=100,
    long_period=200,
    stop_loss = 0,        # 2% stop loss, 0 if we do not want it, try it with 0.02
    take_profit=0,      # 5% take profit, 0.05
    trailing_stop=0.1,     # 2% trailing stop, 0.02
    initial_capital=100000,
    slippage= 0,  #0.001,
    commission=0,  #0.001 - bossa 0.38%
    percents=20 # percent of capital
)
