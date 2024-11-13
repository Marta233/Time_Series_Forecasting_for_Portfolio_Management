import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from datetime import datetime
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from plotly.io import show

# Define custom colors
colors = ["#283149", "#404B69", "#DBEDF3", "#DBDBDB", "#FFFFFF"]
colors2 = ["#FFFFFF", "#DBDBDB", "#DBEDF3", "#404B69", "#283149"]
colors3 = ['#404B69', '#5CDB95', '#ED4C67', '#F7DC6F']

# Class for Portfolio Analysis
class PortfolioAnalysis:
    def __init__(self, tickers, start_date='2015-01-01', end_date='2024-10-31'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.close_prices = None
        self.log_returns = None
        self.correlation_matrix = None
        self.covariance_matrix = None
        self.yearly_returns = None

    def fetch_data(self):
        """Fetch stock data for the selected tickers."""
        for symbol in self.tickers:
            try:
                self.stock_data[symbol] = yf.download(symbol, start=self.start_date, end=self.end_date)
                print(f"Data successfully retrieved for {symbol}.")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    def prepare_data(self):
        """Prepare the data by concatenating 'Adj Close' prices and calculating log returns."""
        # Concatenate 'Close' prices
        self.close_prices = pd.concat([data['Adj Close'] for data in self.stock_data.values()], axis=1)
        self.close_prices.columns = self.tickers

        # Calculate log returns
        self.log_returns = np.log(1 + self.close_prices.pct_change())

    def summarize_data(self):
        """Generate a summary of descriptive statistics."""
        summary_data = []
        for symbol, data in self.stock_data.items():
            summary = [symbol] + list(data['Adj Close'].describe().values)
            summary_data.append(summary)
        headers = ['Stock Symbol', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        print(tabulate(summary_data, headers=headers))

    def calculate_correlation_and_covariance(self):
        """Calculate and display correlation and covariance matrix."""
        self.covariance_matrix = self.log_returns.cov()
        self.correlation_matrix = self.log_returns.corr()

        print("Covariance Matrix:")
        print(self.covariance_matrix)

        print("Correlation Matrix:")
        print(self.correlation_matrix)

    def plot_log_returns(self):
        """Plot log returns for each stock symbol."""
        plt.figure(figsize=(12, 8))
        for symbol in self.log_returns.columns:
            plt.plot(self.log_returns.index, self.log_returns[symbol], label=symbol)
        plt.title('Log Returns for All Symbols')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.grid(True)
        plt.legend(title='Symbols', loc='upper right')

        # Show highest and lowest returns
        highest_point = self.log_returns.max().max()
        lowest_point = self.log_returns.min().min()
        idx_highest = self.log_returns.stack().idxmax()
        idx_lowest = self.log_returns.stack().idxmin()

        plt.text(0.05, 0.95, f'Highest Return:\nDate: {idx_highest[0]}\nSymbol: {idx_highest[1]}\nValue: {highest_point:.2f}',
                 transform=plt.gca().transAxes, color='blue', ha='left', va='top')
        plt.text(0.05, 0.85, f'Lowest Return:\nDate: {idx_lowest[0]}\nSymbol: {idx_lowest[1]}\nValue: {lowest_point:.2f}',
                 transform=plt.gca().transAxes, color='red', ha='left', va='top')
        plt.show()

    def portfolio_optimization(self):
        """Perform portfolio optimization using random portfolio generation."""
        # Calculate expected returns and covariance matrix
        expected_returns = self.log_returns.mean()
        covariance_matrix = self.log_returns.cov()

        # Generate random portfolios
        num_assets = len(self.close_prices.columns)
        num_portfolios = 10000
        p_ret, p_vol, p_weights = [], [], []

        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize weights
            p_weights.append(weights)
            returns = np.dot(weights, expected_returns)
            p_ret.append(returns)
            variance = covariance_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
            sd = np.sqrt(variance)
            ann_sd = sd * np.sqrt(250)
            p_vol.append(ann_sd)

        # Create DataFrame for portfolios
        portfolios = pd.DataFrame({'Returns': p_ret, 'Volatility': p_vol})
        for counter, symbol in enumerate(self.close_prices.columns.tolist()):
            portfolios[symbol + ' weight'] = [w[counter] for w in p_weights]
        
        return portfolios

    def plot_efficient_frontier(self, portfolios):
        """Plot the efficient frontier."""
        portfolios.plot.scatter(x='Volatility', y='Returns', s=20)

        # Finding the optimal portfolio
        min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
        rf = 0.01  # risk-free rate
        optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - rf) / portfolios['Volatility']).idxmax()]

        # Plotting the optimal portfolio
        plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3, label='Portfolios')
        plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500, label='Minimum Volatility Portfolio')
        plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500, label='Optimal Risky Portfolio')
        plt.xlabel('Volatility')
        plt.ylabel('Returns')
        plt.title('Optimal Portfolio vs. Other Portfolios')
        plt.legend()
        plt.show()

    def model_training(self):
        """Train a Mean-Variance model and plot performance."""
        prices = self.close_prices
        X = prices_to_returns(prices)
        X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)

        model = MeanRisk(risk_measure=RiskMeasure.VARIANCE, efficient_frontier_size=30, portfolio_params=dict(name="Variance"))
        model.fit(X_train)

        population_train = model.predict(X_train)
        population_test = model.predict(X_test)

        population_train.set_portfolio_params(tag="Train")
        population_test.set_portfolio_params(tag="Test")
        population = population_train + population_test

        # Plot measures
        fig = population.plot_measures(x=RiskMeasure.ANNUALIZED_VARIANCE, y=PerfMeasure.ANNUALIZED_MEAN, color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
                                       hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO])
        show(fig)

        population_train.plot_composition()
        population_test.measures(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
        population.summary()


