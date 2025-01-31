# Project 2 - Summary

## Itinial idea

In our first project, we developed three strategies based on technical analysis and one based on fundamental analysis. For this project, we decided to focus on two strategies—Trend-Following and Trend-Reversal. We used Random Forest to enhance both strategies by making predictions based on technical indicators. However, due to challenges aligning the VIX with random time series in the Trend-Reversal strategy, we decided to concentrate on further enhancing the Trend-Following strategy.


## Strategies recap 
**Trend-Reversal Strategy**
The trend reversal strategy identifies potential market turning points by analyzing divergences in price and momentum (MACD), overbought or oversold conditions (RSI), and market volatility (VIX).
1.	A Buy Signal is generated when RSI is below 30 (oversold), MACD shows bullish divergence (momentum rising while price is falling), and VIX is higher than its rolling average.
2.	A Sell Signal occurs when RSI exceeds 70 (overbought), MACD shows bearish divergence (momentum falling while price is rising), and VIX is below its rolling average.
The economic reasoning for this strategy includes:

•	RSI identifies overbought or oversold market conditions, often leading to reversals.

•	MACD divergence detects shifts in momentum before prices adjust.

•	VIX compares volatility to its average, signaling potential reversals during heightened or reduced market stress.

**Trend-Following Strategy**
This strategy identifies buying and selling opportunities by using three key indicators: MACD, RSI, and ATR.
1.	A Buy Signal is triggered when the MACD is bullish (above the Signal Line), RSI is above 50, and ATR is higher than its 30-day rolling average.
2.	A Sell Signal occurs when the MACD is bearish (below the Signal Line), RSI is below 50, and ATR is still above the rolling average.
The economic reasoning behind this strategy lies in the role of each indicator:

•	MACD highlights the momentum and direction of the trend. When MACD crosses above the Signal Line, it suggests buying pressure; when it crosses below, it indicates selling pressure.

•	RSI confirms the strength of the trend, helping to filter out weaker signals.

•	ATR ensures there’s enough market volatility to sustain the trend rather than reversing it.


*Both strategies showed similar results in our tests, often with negative Sharpe ratios, meaning poor risk-adjusted returns. However, while Trend-Following was adaptable across time series, the Trend-Reversal strategy required careful alignment with the VIX, making it less practical for this project. As a result, we focused on enhancing the Trend-Following strategy.*

## Updated (and final) Trend-Following strategy and economic reasoning
The final strategy builds on the foundational indicators we used in earlier projects, including MACD, RSI, ATR, and two others (ADX and OBV), but also uses Random Forest to combine these indicators into a predictive framework.

**Indicators used:**

**MACD** (Moving Average Convergence Divergence) - it compares two exponential moving averages (fast and slow) to identify trend direction and momentum.

• Buy Signal: MACD crosses above its signal line.

• Sell Signal: MACD crosses below its signal line.

**RSI** (Relative Strength Index) - measures the speed and change of price movements to identify overbought or oversold conditions. The thresholds used (55 and 45) are less aggressive than the classic 70/30, allowing the strategy to respond to subtler trend changes.

• Buy Signal: RSI > 55 (indicates bullish sentiment).

• Sell Signal: RSI < 45 (indicates bearish sentiment).


**ATR** (Average True Range) - measures market volatility to ensure the trend has enough strength.

• High volatility signal: ATR > 30-day rolling mean ATR.

• Neutral signal: ATR ≤ 30-day rolling mean ATR.

**ADX** (Average Directional Index) - differentiates between trending and ranging markets to ensure signals align with strong trends.

• Strong trend signal: ADX > 25.

• Neutral signal: ADX ≤ 25.


**OBV** (On-Balance Volume) - tracks volume flow to confirm price trends.

• Increasing volume (bullish) (+1): OBV > OBV(-1).

• Neutral or bearish signal (0): OBV ≤ OBV(-1).


**Random Forest integration.**
It predicts future market movements based on the combined signals from all the selected indicators. Instead of relying solely on fixed rule-based conditions, the model learns patterns from historical data to classify future price movements into three categories:

Buy: If the price is expected to increase by more than 5%.

Sell: If the price is expected to decrease by more than 10%.

Hold: If no significant movement is expected.

The model works by aggregating signals from MACD, RSI, ATR, ADX, and OBV into a structured feature set. The target variable is defined based on future price changes, classifying whether the price will increase, decrease, or stay neutral. The model is trained on a rolling window of 25 recent observations, continuously adjusting to the latest market dynamics. 

## Results

| Filename        | Return %     | Max Drawdown %          | Sharpe Ratio |
|-----------------|--------------|-------------------------|-------------|
| btc_v_w     | 158.4            | 8.46                    | 0.51        |
| ge_us_m     | 70.49            | 5.66                    | -1.25       |
| usdjpy_w    | 2.88             | 2.3                     | -6.64       |
| wig20_w     | 30.41            | 2.87                    | -0.8        |
| zw=f_copper | 28.07            | 11.59                   | -0.7        |


1) BTC (Bitcoin) - shows the highest return (158.4%) among all tested assets, indicating that the strategy captured significant upward trends in the cryptocurrency market. Despite the high return, the drawdown remained relatively moderate (8.46%), reflecting controlled risk during downward movements. A positive Sharpe ratio indicates that the strategy provided excess returns relative to its risk for BTC, suggesting successful application of the strategy on Bitcoin.

2) General Electric - strategy showed a solid return (70.49%), meaning the strategy is able to capitalize on trends within the stock market. The drawdown was low, showcasing good risk management for this asset. Despite a good return, the negative Sharpe ratio implies that the risk-adjusted performance was poor.

3) USD/JPY - the strategy struggled to achieve significant returns in the foreign exchange market, indicating that the trends in this pair might not align well with the strategy's parameters. The low drawdown suggests minimal risk exposure but also limited profit opportunities. And, a negative Sharpe (-6.64) ratio indicates poor risk-adjusted performance.

4) WIG20 index - A moderate return (30.41%), suggesting that the strategy was able to capture some trends in the Polish stock market. A low drawdown shows good control over downside risk. The negative Sharpe ratio suggests that, while the return was decent, the strategy did not perform well when adjusted for risk.

5) Copper Futures - Similar to WIG20, the strategy achieved moderate success in identifying trends within the commodity market. The highest drawdown among the assets tested, indicating higher exposure to risk. A negative Sharpe ratio shows that the risk-adjusted returns also were not great.

