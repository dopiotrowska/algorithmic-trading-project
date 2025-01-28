import backtrader as bt
import yfinance as yf
import datetime

class TrendFollowing(bt.Strategy):
    def __init__(self, stop_loss, take_profit, trailing_stop):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop

        # Indicators
        self.macd = bt.indicators.MACD(self.data.close)
        self.rsi = bt.indicators.RSI(self.data.close)
        self.atr = bt.indicators.ATR(self.data)
        self.atr_rolling_mean = bt.indicators.SimpleMovingAverage(self.atr, period=30)

        # To track the entry price of the trade
        self.entry_price = None

    def next(self):
        # Ensure enough data is available (check for ATR initialization)
        if len(self.data) < 30:
            return

        # Check if the number of open positions is less than 10
        if len(self.broker.positions) < 10:
            # Buy signal: MACD bullish, RSI > 50, and ATR is high
            if (self.macd.macd[0] > self.macd.signal[0] and 
                self.rsi[0] > 50 and 
                self.atr[0] > self.atr_rolling_mean[0] and 
                not self.position):
                print(f"Buy signal at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.buy()  # Place a buy order
                self.entry_price = self.data.close[0]  # Record the entry price

            # Sell signal: MACD bearish, RSI < 50, and ATR is high
            elif (self.macd.macd[0] < self.macd.signal[0] and 
                  self.rsi[0] < 50 and 
                  self.atr[0] > self.atr_rolling_mean[0] and 
                  self.position):
                print(f"Sell signal at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.sell()  # Place a sell order

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
def run_backtest(symbol, start_date, end_date, interval, stop_loss, take_profit, trailing_stop, initial_capital=100000, slippage=0.002, commission=0.004, percents=10):
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Fetch data from Yahoo Finance using yfinance
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

    # Print the number of observations in the dataset
    print(f"Number of observations in the dataset: {len(data)}")

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
    cerebro.addstrategy(TrendFollowing, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)

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

    # Print performance metrics
    final_portfolio_value = cerebro.broker.getvalue()
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

# Example usage of the method with custom parameters
symbol = "AAPL"  
start_date = "2023-01-21"
end_date = "2025-01-23"
interval = "1d" 

run_backtest(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    interval=interval,
    stop_loss=0,
    take_profit=0,
    trailing_stop=0,
    initial_capital=100000,
    slippage=0.002,  # 0.2% slippage
    commission=0.004,  # 0.4% commission
    percents=10  # Max 10% of capital per trade
)
