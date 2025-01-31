import backtrader as bt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV

class OBV(bt.Indicator):
    lines = ('obv',)
    params = dict()

    def __init__(self):
        self.addminperiod(2)  # Minimum period required to calculate OBV

    def next(self):
        if self.data.close[0] > self.data.close[-1]:  # Price up
            self.lines.obv[0] = self.lines.obv[-1] + self.data.volume[0]
        elif self.data.close[0] < self.data.close[-1]:  # Price down
            self.lines.obv[0] = self.lines.obv[-1] - self.data.volume[0]
        else:  # Price unchanged
            self.lines.obv[0] = self.lines.obv[-1]


class TrendFollowingWithRF(bt.Strategy):
    def __init__(self, macd_fast_period=12, macd_slow_period=26, macd_signal_period=9,
                 rsi_period=14, atr_period=14, stop_loss=0.02, take_profit=0.05,
                 trailing_stop=0.02, signal_window=25, prediction_window=10):

        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.signal_window = signal_window
        self.prediction_window = prediction_window

        # Indicators (MACD, RSI, ATR, ATR rolling mean)
        self.macd = bt.indicators.MACD(self.data.close, 
                                       period_me1=self.macd_fast_period,
                                       period_me2=self.macd_slow_period, 
                                       period_signal=self.macd_signal_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.rsi_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.atr_period)
        self.atr_rolling_mean = bt.indicators.SimpleMovingAverage(self.atr, period=30)
        self.adx = bt.indicators.ADX(self.data, period=14)
        self.bollinger = bt.indicators.BollingerBands(self.data, period=20)
        self.obv = OBV(self.data)

        # Random Forest Model
        self.rf_model = RandomForestClassifier(
            n_estimators=500, 
            max_depth=10, 
            min_samples_leaf=5, 
            random_state=42
        )

        # Data collection for model training
        self.features = []  # Features (signals)
        self.target = []  # Target (future price movement)

        # Track entry price and stop-loss levels
        self.entry_price = None

    def next(self):
        # Ensure enough data for indicators
        if len(self.data) < max(self.atr_period, self.prediction_window):
            return

        # Current index or position in the data array
        current_index = len(self)

        # Number of data points remaining to analyze
        remaining_data_points = self.data.buflen() - current_index
        if remaining_data_points <= self.prediction_window:
            print("Not enough data left to make a prediction")
            return

        # Generate Binary Signals (Based on Indicators)
        macd_signal = 1 if self.macd.macd[0] > self.macd.signal[0] else -1 if self.macd.macd[0] < self.macd.signal[0] else 0
        rsi_signal = 1 if self.rsi[0] > 55 else -1 if self.rsi[0] < 45 else 0
        atr_signal = 1 if self.atr[0] > self.atr_rolling_mean[0] else 0
        adx_signal = 1 if self.adx[0] > 25 else 0
        volume_signal = 1 if self.obv[0] > self.obv[-1] else 0  

        # Combine Signals for Prediction
        signal_values = [macd_signal, rsi_signal, atr_signal, adx_signal, volume_signal]
        self.features.append(signal_values)

        # Generate Target
        pct_change = (self.data.close[self.prediction_window] - self.data.close[0]) / self.data.close[0]
        if pct_change > 0.05:
            self.target.append(1)  # Buy signal
        elif pct_change < -0.1:
            self.target.append(-1)  # Sell signal
        else:
            self.target.append(0)  # No signal

        # Train the Random Forest Model
        if len(self.features) >= self.signal_window:
            X = np.array(self.features[-self.signal_window:])
            y = np.array(self.target[-self.signal_window:])
            if len(set(y)) > 1:  # Ensure variability in target

                # param_grid = {
                #    'n_estimators': [100, 200, 500],
                #    'max_depth': [5, 10, 15],
                #    'min_samples_leaf': [3, 5, 7]
                # }
                # grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
                # grid_search.fit(X, y)
                
                self.rf_model.fit(X, y)
                
                prediction = self.rf_model.predict([signal_values])[0]

                # Act based on the prediction
                if prediction == 1 and not self.position:
                    self.buy_signal()
                elif prediction == -1 and not self.position:
                    self.sell_signal()
                elif prediction == 1 and self.position.size < 0:
                    self.close()
                    self.buy_signal()
                elif prediction == -1 and self.position.size > 0:
                    self.close()
                    self.sell_signal()

        # Check stop-loss
        if self.position:
            self.check_stop_loss()


    def buy_signal(self):
       size = self.broker.get_cash() * 0.1 / self.data.close[0]  # 10% of capital per trade
       print(f"Buy signal at {self.data.datetime.datetime()} with size {size}")
       self.buy(size=size)
       self.entry_price = self.data.close[0]

    def sell_signal(self):
       size = self.broker.get_cash() * 0.1 / self.data.close[0]  # 10% of capital per trade
       print(f"Sell signal at {self.data.datetime.datetime()} with size {size}")
       self.sell(size=size)
       self.entry_price = self.data.close[0]

    def check_stop_loss(self):
        if self.stop_loss != 0:
            atr_sl = self.atr[0] * 1.5  # Use ATR-based stop-loss instead of fixed %
            stop_loss_price = self.entry_price - atr_sl if self.position.size > 0 else self.entry_price + atr_sl
            #stop_loss_price = self.entry_price * (1 - self.stop_loss if self.position.size > 0 else 1 + self.stop_loss)
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


def run_backtest(symbol=None, start_date=None, end_date=None, interval=None, stop_loss=0.02, take_profit=0.05, trailing_stop=0.02, csv_path='', separator=',', initial_capital=100000, slippage=0.002, commission=0.004):

    cerebro = bt.Cerebro()

    try:
        if csv_path:
            data = pd.read_csv(csv_path, index_col='Date', parse_dates=True, sep=separator)
            data.rename(columns={'Min': 'Low', 'Max': 'High'}, inplace=True)
            print(f"Data loaded from {csv_path}")
        elif symbol and start_date and end_date and interval:
            # Fetch data from Yahoo Finance using yfinance
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

            # Check if data is empty
            if data.empty:
                print(f"No data downloaded for {symbol} from {start_date} to {end_date} with interval {interval}")
                return
        else:
            print("Either provide a CSV path or Yahoo Finance parameters (symbol, start_date, end_date, interval).")
            return

        # Print the number of observations in the dataset
        print(f"Number of observations in the dataset: {len(data)}")

        # Load data into Backtrader
        data_feed = bt.feeds.PandasData(dataname=data)

        # Add the data to the Cerebro engine
        cerebro.adddata(data_feed)

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Set initial capital
    cerebro.broker.set_cash(initial_capital)

    # Set commission and slippage
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)

    # Print the starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")

    # Add the strategy to the Cerebro engine, passing the parameters
    cerebro.addstrategy(
        TrendFollowingWithRF,
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
    return_percentage = round(((final_portfolio_value - initial_capital) / initial_capital) * 100, 2)
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Return in %: {return_percentage:.2f}%")
    if sharpe_ratio is not None:
        sharpe_ratio = round(sharpe_ratio, 2)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Sharpe Ratio: None")
    max_drawdown = round(drawdown.max.drawdown, 2)
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print()

    # Append results to CSV
    results_data = {
        'filename': csv_path.split('/')[-1] if csv_path else f"{symbol}_{start_date}_{end_date}",
        'return_percentage': return_percentage,
        'max_drawdown_percentage': max_drawdown,
        'sharpe_ratio': sharpe_ratio if sharpe_ratio is not None else 'None'
    }
    results_df = pd.DataFrame([results_data])
    results_df.to_csv('Project2/results2.csv', mode='a', header=False, index=False)

    # Plot the results
    cerebro.plot()

# Example usage
# symbol = "AAPL"  
# start_date = "2024-01-21"
# end_date = "2025-01-23"
# interval = "1d" 
csv_path = "Project2/data/zw=f_copper.csv"

run_backtest(
    # symbol=symbol,
    # start_date=start_date,
    # end_date=end_date,
    # interval=interval,
    stop_loss=0.05,
    take_profit=0.1,
    trailing_stop=0.05,
    csv_path=csv_path,
    separator=';',
    initial_capital=100000,
    slippage=0.002,  # 0.2% slippage
    commission=0.004  # 0.4% commission
)
