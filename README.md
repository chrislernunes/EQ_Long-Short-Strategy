# EQ_Long-Short-Strategy

Abstract

This trading strategy aims to capitalize on market trends using a combination of technical indicators, including Bollinger Bands, MACD (Moving Average Convergence Divergence), and Ichimoku Cloud components (Tenkan-sen and Kijun-sen). The strategy utilizes Buy signals derived from the intersection of these indicators to initiate and exit trades.
The algorithm identifies entry points by detecting when the closing price exceeds the upper Bollinger Band, the MACD surpasses the Signal Line, and the Tenkan-sen is above the Kijun-sen. Subsequently, the strategy triggers a Buy signal. Exit points are determined when the closing price fails to satisfy the entry conditions, leading to the execution of a Exit signal.
Key performance metrics, such as accuracy rate, total number of trades, winning and losing trades, win/loss ratio, profit factor, max loss, max gain, Sharpe ratio, max drawdown, and corresponding periods, are calculated to assess the effectiveness of the strategy.
