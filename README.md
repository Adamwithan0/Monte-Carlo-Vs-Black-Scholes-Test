# Monte-Carlo-Vs-Black-Scholes-Test

 ![Monte Carlo](Monte%20Carlo%20vs%20Black-Scholes/plot.png)


## Introduction

In this coding project, we delve into the domain of option pricing using two powerful methods - the Black-Scholes formula and Monte Carlo simulation. Our goal is to estimate the price of European call options for five renowned tech stocks, namely AAPL, MSFT, GOOGL, AMZN, and TSLA. By leveraging historical stock price data from January 1, 2020, to January 1, 2021, we'll explore how these financial models can provide valuable insights for traders and investors.

## Data Retrieval

To kick things off, we fetch the historical stock price data for the selected tech stocks. This data serves as the foundation for our analysis, enabling us to understand the behavior and trends of each stock over the specified period. 
```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_annual_return(stock_data):
    price_at_start = stock_data['Adj Close'][0]
    price_at_end = stock_data['Adj Close'][-1]
    annual_return = (price_at_end / price_at_start) - 1
    return annual_return

def get_bond_yield(maturity):
    # Assuming the yield data is available for the U.S. 1-year Treasury bond with ticker symbol "^IRX"
    bond_ticker = "^IRX"
    bond_data = yf.download(bond_ticker)

    # Find the yield for the given maturity (in years)
     # Note: We're assuming the bond_data contains daily yields. You may need to handle different data frequencies.
    bond_yield = bond_data['Adj Close'].iloc[-1] / 100.0  # Adjust for percentage

    return bond_yield

```
## The Black-Scholes Formula

One of the pillars of modern finance, the Black-Scholes formula, comes into play. Using this elegant equation, we calculate the theoretical price of European call options for each stock. Taking into account factors like stock price, strike price, risk-free rate, time to maturity, and stock volatility, the formula provides an estimated option price consistent with modern pricing techniques.

```python

def black_scholes_call_option(S, X, r, T, sigma):
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_option_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_option_price

```

## Monte Carlo Simulation

In addition to the Black-Scholes formula, we employ the versatile Monte Carlo simulation method. This simulation technique involves generating numerous random price paths for each stock and using them to estimate option prices probabilistically. By running 10,000 simulations, we gain a clearer understanding of the potential range of option prices.

```python
def monte_carlo_simulation(S, mu, sigma, T, num_simulations, num_periods):
    dt = T / num_periods
    price_matrix = np.zeros((num_periods + 1, num_simulations))
    price_matrix[0] = S

    for i in range(1, num_periods + 1):
        random_returns = np.random.normal((mu * dt), (sigma * np.sqrt(dt)), num_simulations)
        price_matrix[i] = price_matrix[i - 1] * (1 + random_returns)

    return price_matrix

def monte_carlo_simulation_european_call(S, mu, sigma, X, T, r, num_simulations):
    dt = T / 252  # Assuming 252 trading days in a year
    price_matrix = np.zeros((252 + 1, num_simulations))
    price_matrix[0] = S

    for i in range(1, 252 + 1):
        random_returns = np.random.normal((mu * dt), (sigma * np.sqrt(dt)), num_simulations)
        price_matrix[i] = price_matrix[i - 1] * (1 + random_returns)

    final_stock_price = price_matrix[-1]
    option_payoffs = np.maximum(final_stock_price - X, 0)
    option_price = np.mean(option_payoffs) * np.exp(-r * T)

    return option_price
```

## Option Price Comparison

With both pricing methods in hand, we proceed to compare their respective option prices for each stock. This comparison sheds light on the differences and similarities between the two approaches, offering valuable insights for traders and investors seeking the most suitable pricing strategy.

## Visual Representation

To make the findings more accessible, we create informative bar charts that showcase the comparison between the Black-Scholes and Monte Carlo option prices. These visual representations provide a clear and concise summary of our analysis.

```python
def plot_simulations(simulated_prices, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_prices[:, :10])
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Simulation for {ticker} Forecast")
    plt.grid(True)
    plt.savefig(f"{ticker}_plot.png")
    plt.close()
```

## Conclusion

As we conclude our exploration into Monte Carlo option pricing, we unveil a nuanced understanding of the two pricing methods. The Pricing varied drastically using these two methods, demonstrating the discrepancies between them. Namely that the expected growth of these tech stocks, was likely overestimated, given their high recent annual growth, resulting in a large upswing in price according to our Monte 
Carlo simulation. This had the result of the option price being calculated at a far higher rate than as predicted by Black-Scholes, making this a poor instance to use a Monte Carlo method.

## Requirements

To run the script, you'll need the following Python libraries:
- yfinance
- numpy
- matplotlib
- scipy

















