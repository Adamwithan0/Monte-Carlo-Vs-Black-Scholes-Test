# Monte-Carlo-Vs-Black-Scholes-Test

 ![Monte Carlo](Monte%20Carlo%20vs%20Black-Scholes/plot.png)


## Introduction

In this coding project, we delve into the domain of option pricing using two powerful methods - the Black-Scholes formula and Monte Carlo simulation. Our goal is to estimate the price of European call options for five renowned tech stocks, namely AAPL, MSFT, GOOGL, AMZN, and TSLA. By leveraging historical stock price data from January 1, 2020, to January 1, 2021, we'll explore how these financial models can provide valuable insights for traders and investors.

## Data Retrieval

To kick things off, we fetch the historical stock price data for the selected tech stocks. This data serves as the foundation for our analysis, enabling us to understand the behavior and trends of each stock over the specified period. 

## The Black-Scholes Formula

One of the pillars of modern finance, the Black-Scholes formula, comes into play. Using this elegant equation, we calculate the theoretical price of European call options for each stock. Taking into account factors like stock price, strike price, risk-free rate, time to maturity, and stock volatility, the formula provides an estimated option price consistent with modern pricing techniques.

## Monte Carlo Simulation

In addition to the Black-Scholes formula, we employ the versatile Monte Carlo simulation method. This simulation technique involves generating numerous random price paths for each stock and using them to estimate option prices probabilistically. By running 10,000 simulations, we gain a clearer understanding of the potential range of option prices.

## Option Price Comparison

With both pricing methods in hand, we proceed to compare their respective option prices for each stock. This comparison sheds light on the differences and similarities between the two approaches, offering valuable insights for traders and investors seeking the most suitable pricing strategy.

## Visual Representation

To make our findings more accessible, we create informative bar charts that showcase the comparison between the Black-Scholes and Monte Carlo option prices. These visual representations provide a clear and concise summary of our analysis.

## Conclusion

As we conclude our exploration into Monte Carlo option pricing, we unveil a nuanced understanding of the two pricing methods. Armed with insights from the Black-Scholes formula and the Monte Carlo simulation, traders and investors can make informed decisions regarding their option trading strategies.

## Requirements

To run the script, you'll need the following Python libraries:
- yfinance
- numpy
- matplotlib
- scipy


