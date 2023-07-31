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

def black_scholes_call_option(S, X, r, T, sigma):
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_option_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_option_price

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


def plot_simulations(simulated_prices, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_prices[:, :10])
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Simulation for {ticker} Forecast")
    plt.grid(True)
    plt.savefig(f"{ticker}_plot.png")
    plt.close()


def get_bond_yield(maturity):
    # Assuming the yield data is available for the U.S. 1-year Treasury bond with ticker symbol "^IRX"
    bond_ticker = "^IRX"
    bond_data = yf.download(bond_ticker)

    # Find the yield for the given maturity (in years)
     # Note: We're assuming the bond_data contains daily yields. You may need to handle different data frequencies.
    bond_yield = bond_data['Adj Close'].iloc[-1] / 100.0  # Adjust for percentage

    return bond_yield



if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # List of stock tickers to compare
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    num_simulations = 10000
    num_periods = 252

    for ticker in tickers:
        stock_data = get_stock_data(ticker, start_date, end_date)
        annual_return = calculate_annual_return(stock_data)

        S = stock_data['Adj Close'][-1]
        mu = annual_return
        sigma = np.std(stock_data['Adj Close'].pct_change())
        T = 1

        # Get the current bond yield for a risk-free rate with a similar maturity as the option
        option_maturity = T  # Assuming the option maturity is the same as T
        r = get_bond_yield(option_maturity)

        # Calculate theoretical option price using Black-Scholes formula
        X = 200  # Strike price (you can change this to any desired value)
        theoretical_option_price = black_scholes_call_option(S, S, r, T, sigma)

        # Perform Monte Carlo simulation to approximate the option price
        monte_carlo_option_price = monte_carlo_simulation_european_call(S, mu, sigma, S, T, r, num_simulations)

        # Visualize the comparison between Black-Scholes and Monte Carlo option prices
        plt.figure(figsize=(8, 6))
        plt.bar(['Black-Scholes', 'Monte Carlo'], [theoretical_option_price, monte_carlo_option_price])
        plt.xlabel('Option Pricing Method')
        plt.ylabel('Option Price')
        plt.title(f'European Call Option Price Comparison for {ticker}')
        plt.savefig(f'{ticker}_option_comparison.png')
        plt.close()

        print(f"Stock: {ticker}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Volatility: {sigma:.2%}")
        print(f"Risk-Free Rate: {r:.2%}")
        print(f"Theoretical Option Price: ${theoretical_option_price:.2f}")
        print(f"Monte Carlo Option Price: ${monte_carlo_option_price:.2f}")