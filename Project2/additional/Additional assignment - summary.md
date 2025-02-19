# Additional assignment - project 2 - Summary

The base strategy for this assignment was taken from Project 2—a trend-following strategy using Random Forest to generate trading signals. My objective was to optimize and improve the strategy by testing different machine learning models, indicator combinations, and risk control mechanisms.

## Testing different machine learning models

The first step in optimization focused on improving the machine learning model. The number of trees in the Random Forest was reduced from 500 to 200 to see if this would make the model run faster without affecting performance. The results showed almost no change, meaning reducing the number of trees did not impact predictions.

LightGBM was also tested as a faster alternative. It improves decision trees using Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) (Ke et al., 2017), making it more efficient for large datasets. However, despite adjusting its hyperparameters, LightGBM did not produce meaningful results, so I reverted to Random Forest as the final model.


## Experimenting with different indicators

Next, the strategy was tested with various technical indicators to evaluate their impact on performance. Bollinger Bands were included, as they were previously used in Project 1, and testing them again allowed for a comparison of their effectiveness in this setup. Additionally, the Stochastic Oscillator was introduced after seeing it in another group’s presentation, where it appeared to be a useful tool for identifying overbought and oversold conditions.

Other indicators, such as the Exponential Smoothing Line (EMA Signal), were also tested, but it did not provide any noticeable improvements. Similarly, ADX (Average Directional Index) and OBV (On-Balance Volume) were analyzed but did not significantly enhance performance. As a result, only the most effective indicators were retained: MACD, RSI, ATR, Bollinger Bands, and the Stochastic Oscillator.

Reasoning to include each indicator:

* MACD (Moving Average Convergence Divergence): Identifies trend direction and momentum shifts through moving average crossovers. It helps confirm when a new trend is forming.

* RSI (Relative Strength Index): Measures the strength of price movements to determine if an asset is overbought or oversold. In this strategy case, RSI confirms the strength of the trend, helping to filter out weaker signals.

* ATR (Average True Range): Captures market volatility.

* Bollinger Bands: Identify volatility and potential price breakouts, assisting in trend confirmation when combined with MACD and RSI.

* Stochastic Oscillator: Improves entry and exit timing by identifying price momentum relative to recent price ranges, helping to catch trend reversals earlier.

By combining these indicators, the strategy was aimed to balance trend identification (MACD, Bollinger Bands), momentum confirmation (RSI, Stochastic), and volatility adjustments (ATR).

## Adjusting indicator parameters for faster signal capture

One major issue observed was that the strategy often entered and exited trades too late. To improve responsiveness, several adjustments were made:

* Shortened the MACD periods for faster trend detection.
* Reduced the Bollinger Bands period to capture breakouts sooner.
* Decreased the Stochastic Oscillator period to generate quicker overbought/oversold signals.

 
These changes improved entry and exit timing, making the strategy more reactive to price movements for BTC (Bitcoin) and GE (General Electric), but not did not work effectively for other time series.

To allow for a more detailed comparison, both versions of the strategy were retained:

* Results1.csv and Results3.csv containe performance metrics for the shorter-period indicators, which showed improved responsiveness for BTC and GE.
* Results2.csv and Results4.csv include metrics for the original indicator settings, providing a reference for how the strategy performs under normal conditions.


## Optimizing risk control mechanisms

Finally, different risk management techniques were tested to improve capital protection and optimize trade exits. The first approach focused solely on a Dynamic ATR-based Stop Loss, which adjusts to market volatility rather than using a fixed percentage. This method ensures that stop-loss levels are wider in high-volatility conditions and tighter in low-volatility conditions. 

The results for this approach are recorded in Results1.csv and Results2.csv, corresponding to the strategy versions with shorter and normal indicator periods.

Next, Take Profit and Trailing Stop were tested individually alongside the Dynamic ATR-based Stop Loss. However, adding only one of these controls did not lead to significant improvements in strategy performance. To address this, all three risk management techniques were used together—ATR-based Stop Loss, Take Profit, and Trailing Stop—to create a more adaptive and flexible approach. This combination helped to protect gains while ensuring that losses remained controlled, especially in volatile market conditions. 

The results for this final version of the strategy are stored in Results3.csv and Results4.csv, reflecting both the shorter and normal indicator periods.


## Results

### Results1.csv vs. Results2.csv (impact of shorter indicator periods)

BTC & GE: The adjusted version (Results1.csv) significantly improved returns for Bitcoin (40,670.93% vs. 21,961.31%) and General Electric (480.89% vs. 301.55%). However, Sharpe Ratios remained low, but still better in results1 (shorter periods) (0.36 vs. 0.32 for BTC, -0.02 vs. -0.15 for GE), indicating that while returns increased, the risk-adjusted performance did not significantly improve.

Other assets: For other time series, shorter periods worsened results. For example, USD/JPY had a drop in return (1.19% vs. 11.45%), while its Sharpe Ratio remained deeply negative (-2.37 vs. -2.22). Similarly for Wig20 and Copper.



### Results1.csv vs. Results3.csv (impact of risk controls)

Adding Take Profit and Trailing Stop (results3.csv) alongside the Dynamic ATR-based Stop Loss (results1.csv) significantly reduced returns but improved risk control.

* Returns dropped sharply, especially for BTC (40,670% → 131%) and GE (480% → 47%), indicating that the new risk controls cut profits early.

* Drawdowns decreased, with BTC’s max drawdown falling from 51% to 10%, showing better capital protection.

* Sharpe Ratio improved for BTC (0.36 → 0.67), but worsened for all other assets, suggesting that these controls help high-volatility assets but limit gains in lower-volatility markets.



### Results2.csv vs. Results4.csv (impact of risk controls)

* Returns dropped significantly across all assets. BTC and GE saw particularly large declines, suggesting that setting profit-taking thresholds limited potential upside. Other assets had much lower returns as well, implying that premature exits may have cut trends short.

* Drawdowns were significantly reduced in BTC, GE, and Copper, indicating improved downside protection. This suggests that additional exit strategies helped control risk exposure.

* For BTC, the Sharpe Ratio improved slightly. For other assets, Sharpe Ratios worsened, indicating that while losses were better controlled, returns suffered disproportionately.





Best Sharpe Ratio for Each Asset (across all cases):

BTC: Best in Results 3 (0.67)

GE: Best in Results 1 (-0.02)

USD/JPY, WIG20, Copper: Best in Results 2 (-2.22, -0.6, 0.38)



### Comparing Results 2 to Project 2 final results

BTC: Project 2 had a better Sharpe Ratio (0.51 vs. 0.32). However, assignment’s Results 2 had an extreme increase in return (21,961.31% vs. 158.4%) at the cost of much higher drawdown (44.84% vs. 8.46%).

GE: Results 2 outperformed Project 2 in Sharpe Ratio (-0.15 vs. -1.25) and return (301.55% vs. 70.49%), but with a significantly higher drawdown (18.5% vs. 5.66%).

USD/JPY, WIG20, and Copper: Results 2 outperformed Project 2 in all aspects (higher return, better Sharpe Ratio, and similar or lower drawdown).


