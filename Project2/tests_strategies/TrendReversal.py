import backtrader as bt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import datetime
import pandas as pd


class TrendReversalWithVIX(bt.Strategy):
    def __init__(self, macd_fast_period=12, macd_slow_period=26, macd_signal_period=9,
                 rsi_period=14, vix_window=20, stop_loss=0.02, take_profit=0.05,
                 trailing_stop=0.02, signal_window=25, prediction_window=10):
        # Initialize parameters
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.rsi_period = rsi_period
        self.vix_window = vix_window
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.signal_window = signal_window
        self.prediction_window = prediction_window

        # Indicators for main asset
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.macd_fast_period,
                                       period_me2=self.macd_slow_period,
                                       period_signal=self.macd_signal_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.rsi_period)

        # VIX rolling mean
        self.vix_rolling_mean = bt.indicators.SimpleMovingAverage(self.datas[1].close, period=self.vix_window)

        # Random Forest Model
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42
        )

        # Data collection for model training
        self.features = []  # Features (signals)
        self.target = []  # Target (future price movement)

        # Track entry price and stop-loss levels
        self.entry_price = None

    def next(self):
        # Ensure enough data for indicators
        if len(self.data) < max(self.rsi_period, self.prediction_window, self.vix_window):
            return

        # Current index or position in the data array
        current_index = len(self)

        # Number of data points remaining to analyze
        remaining_data_points = self.data.buflen() - current_index
        if remaining_data_points <= self.prediction_window:
            print("Not enough data left to make a prediction")
            return

        # Generate signals based on indicators
        macd_divergence_buy = (self.macd.macd[0] > self.macd.macd[-1]) and (self.data.close[0] < self.data.close[-1])
        macd_divergence_sell = (self.macd.macd[0] < self.macd.macd[-1]) and (self.data.close[0] > self.data.close[-1])

        rsi_signal = 1 if self.rsi[0] < 30 else -1 if self.rsi[0] > 70 else 0
        vix_signal = 1 if self.datas[1].close[0] > self.vix_rolling_mean[0] else -1 if self.datas[1].close[0] < self.vix_rolling_mean[0] else 0

        # Combine signals for Random Forest input
        buy_signal = 1 if rsi_signal == 1 and macd_divergence_buy and vix_signal == 1 else 0
        sell_signal = 1 if rsi_signal == -1 and macd_divergence_sell and vix_signal == -1 else 0
        signal_values = [rsi_signal, vix_signal, macd_divergence_buy, macd_divergence_sell]
        self.features.append(signal_values)

        # Generate target based on future price movement
        pct_change = (self.data.close[self.prediction_window] - self.data.close[0]) / self.data.close[0]
        self.target.append(1 if pct_change > 0.02 else -1 if pct_change < -0.02 else 0)

        # Train Random Forest and make predictions
        if len(self.features) >= self.signal_window:
            X = np.array(self.features[-self.signal_window:])
            y = np.array(self.target[-self.signal_window:])
            if len(set(y)) > 1:  # Ensure variability in target
                self.rf_model.fit(X, y)
                prediction = self.rf_model.predict([signal_values])[0]

                # Act based on the prediction
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

    def sell_signal(self):
        print(f"Sell signal at {self.data.datetime.datetime()}")
        self.sell(size=1)
        self.entry_price = self.data.close[0]

    def check_stop_loss(self):
        if self.stop_loss != 0:
            stop_loss_price = self.entry_price * (1 - self.stop_loss if self.position.size > 0 else 1 + self.stop_loss)
            if (self.position.size > 0 and self.data.close[0] < stop_loss_price) or \
               (self.position.size < 0 and self.data.close[0] > stop_loss_price):
                print(f"Stop Loss triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()

        if self.take_profit != 0:
            take_profit_price = self.entry_price * (1 + self.take_profit if self.position.size > 0 else 1 - self.take_profit)
            if (self.position.size > 0 and self.data.close[0] > take_profit_price) or \
               (self.position.size < 0 and self.data.close[0] < take_profit_price):
                print(f"Take Profit triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()

        if self.trailing_stop != 0:
            trailing_stop_price = self.entry_price * (1 - self.trailing_stop if self.position.size > 0 else 1 + self.trailing_stop)
            if (self.position.size > 0 and self.data.close[0] < trailing_stop_price) or \
               (self.position.size < 0 and self.data.close[0] > trailing_stop_price):
                print(f"Trailing Stop triggered at {self.data.datetime.datetime()} for price {self.data.close[0]:.2f}")
                self.close()




def run_backtest(symbol, vix_symbol, start_date, end_date, stop_loss, take_profit, trailing_stop,
                 initial_capital=100000, slippage=0.002, commission=0.004, percents=10):
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Fetch data for the main symbol (e.g., S&P 500 or AAPL)
    asset_data = yf.download(symbol, start=start_date, end=end_date)
    if asset_data.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    # Fetch VIX data
    vix_data = yf.download(vix_symbol, start=start_date, end=end_date)
    if vix_data.empty:
        raise ValueError(f"No data found for VIX symbol: {vix_symbol}")

    # Align both datasets by date
    combined_data = pd.merge(asset_data, vix_data[['Adj Close']], left_index=True, right_index=True, suffixes=('', '_VIX'))
    combined_data.rename(columns={'Adj Close_VIX': 'VIX'}, inplace=True)

    # Add main asset data feed
    asset_data_feed = bt.feeds.PandasData(dataname=asset_data)
    cerebro.adddata(asset_data_feed, name="Main Asset")

    # Add VIX data feed
    vix_data_feed = bt.feeds.PandasData(dataname=vix_data)
    cerebro.adddata(vix_data_feed, name="VIX")

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
    cerebro.addstrategy(
        TrendReversalWithVIX,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop
    )

    # Add analyzers for performance metrics (Sharpe Ratio and Drawdown)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="trans")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run the backtest
    results = cerebro.run()

    # Extract analyzers results
    sharpe_ratio = results[0].analyzers.sharperatio.get_analysis().get('sharperatio', None)
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # Print performance metrics
    final_portfolio_value = cerebro.broker.getvalue()
    return_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print()

    # Plot the results
    cerebro.plot()

# Example usage
symbol = "AAPL"  # Main asset symbol (e.g., S&P 500)
vix_symbol = "^VIX"  # VIX index symbol
start_date = "2022-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

run_backtest(
    symbol=symbol,
    vix_symbol=vix_symbol,
    start_date=start_date,
    end_date=end_date,
    stop_loss=0.02,
    take_profit=0.05,
    trailing_stop=0.02,
    initial_capital=100000,
    slippage=0.002,  # 0.2% slippage
    commission=0.004,  # 0.4% commission
    percents=10  # Max 10% of capital per trade
)

