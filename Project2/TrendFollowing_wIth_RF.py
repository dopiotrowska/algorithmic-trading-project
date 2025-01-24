import backtrader as bt
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime

class TrendFollowingWithRF(bt.Strategy):
    def __init__(self, stop_loss, take_profit, trailing_stop, signal_window=25, prediction_window=10):
        # Initialize parameters
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.signal_window = signal_window
        self.prediction_window = prediction_window

        # Indicators
        self.macd = bt.indicators.MACD(self.data.close)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.atr = bt.indicators.AverageTrueRange(self.data)
        self.atr_rolling_mean = bt.indicators.SimpleMovingAverage(self.atr, period=30)

        # Random Forest Model
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42
        )

        # Data collection for training
        self.features = []
        self.target = []

        # Track position entry and stop-loss levels
        self.entry_price = None
        self.trailing_stop_price = None

    def next(self):
        # Ensure enough data for indicators
        if len(self.data) < max(30, self.prediction_window):
            return

        # Generate features
        macd_diff = self.macd.macd[0] - self.macd.signal[0]
        rsi_value = self.rsi[0]
        atr_diff = self.atr[0] - self.atr_rolling_mean[0]
        signal_values = [macd_diff, rsi_value, atr_diff]
        self.features.append(signal_values)

        # Generate target (future price movement)
        if len(self.data) > self.prediction_window:
            future_pct_change = (self.data.close[self.prediction_window] - self.data.close[0]) / self.data.close[0]
            if future_pct_change > 0.02:
                self.target.append(1)
            elif future_pct_change < -0.02:
                self.target.append(-1)
            else:
                self.target.append(0)

        # Train Random Forest
        if len(self.features) >= self.signal_window:
            X = np.array(self.features[-self.signal_window:])
            y = np.array(self.target[-self.signal_window:])
            if len(set(y)) > 1:  # Ensure variability
                self.rf_model.fit(X, y)
                prediction = self.rf_model.predict([signal_values])[0]

                # Execute trades based on prediction
                if prediction == 1 and not self.position:
                    self.buy_signal()
                elif prediction == -1 and not self.position:
                    self.sell_signal()

        # Manage open positions
        if self.position:
            self.check_stop_loss()

    def buy_signal(self):
        print(f"Buy signal at {self.data.datetime.datetime()}")
        self.buy(size=1)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price * (1 - self.trailing_stop)

    def sell_signal(self):
        print(f"Sell signal at {self.data.datetime.datetime()}")
        self.sell(size=1)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price * (1 + self.trailing_stop)

    def check_stop_loss(self):
        if self.stop_loss:
            stop_loss_price = self.entry_price * (1 - self.stop_loss if self.position.size > 0 else 1 + self.stop_loss)
            if (self.position.size > 0 and self.data.close[0] < stop_loss_price) or \
               (self.position.size < 0 and self.data.close[0] > stop_loss_price):
                print(f"Stop Loss triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()

        if self.take_profit:
            take_profit_price = self.entry_price * (1 + self.take_profit if self.position.size > 0 else 1 - self.take_profit)
            if (self.position.size > 0 and self.data.close[0] > take_profit_price) or \
               (self.position.size < 0 and self.data.close[0] < take_profit_price):
                print(f"Take Profit triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()

        if self.trailing_stop:
            if self.position.size > 0:  # Long position
                self.trailing_stop_price = max(self.trailing_stop_price, self.data.close[0] * (1 - self.trailing_stop))
                if self.data.close[0] < self.trailing_stop_price:
                    print(f"Trailing Stop triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                    self.close()
            elif self.position.size < 0:  # Short position
                self.trailing_stop_price = min(self.trailing_stop_price, self.data.close[0] * (1 + self.trailing_stop))
                if self.data.close[0] > self.trailing_stop_price:
                    print(f"Trailing Stop triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                    self.close()

# Backtest method with custom parameters
def run_backtest(symbol, start_date, end_date, stop_loss, take_profit, trailing_stop, initial_capital=100000, slippage=0.002, commission=0.004, percents=10):
    cerebro = bt.Cerebro()

    # Fetch data using yfinance
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print("No data downloaded. Please check the symbol and date range.")
        return

    # Load data into Backtrader
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set broker settings
    cerebro.broker.set_cash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)

    # Set position size (percent of capital per trade)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=percents)

    # Add the strategy
    cerebro.addstrategy(TrendFollowingWithRF, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    # Print starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run backtest
    results = cerebro.run()

    # Extract and display performance metrics
    sharpe_ratio = results[0].analyzers.sharperatio.get_analysis().get('sharperatio', None)
    drawdown = results[0].analyzers.drawdown.get_analysis()
    final_portfolio_value = cerebro.broker.getvalue()
    return_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100

    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: Not calculated")

    # Plot results
    cerebro.plot()

# Example usage
symbol = "^GSPC"
start_date = "2010-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
run_backtest(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    stop_loss=0.02,
    take_profit=0.05,
    trailing_stop=0.02,
    initial_capital=100000,
    slippage=0.002,
    commission=0.004,
    percents=10
)
