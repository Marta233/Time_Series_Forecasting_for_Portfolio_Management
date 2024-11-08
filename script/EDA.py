import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
class EDA:
    def __init__(self, df, window=30):
        self.df = df
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df['Date'] = self.df['Date'].dt.tz_localize(None)
        self.window = window  # Set the window size for rolling mean and std
    
    def get_summary_stats(self, Ticker_name):
        """
        Computes and returns summary statistics for the DataFrame filtered by Ticker name.
        """
        # Filter the data for the selected ticker
        ticker_data = self.df[self.df['Ticker'] == Ticker_name]
        return ticker_data.describe()

    def missing_percentage(self):
        # Calculate the percentage of missing values
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        
        # Create a DataFrame to display the results nicely
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        
        return missing_df

    def data_types(self):
        data_typs = self.df.dtypes
        return pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': data_typs
        })

    def trend_analysis_close_price(self):
        plt.figure(figsize=(10, 6))
        for ticker in self.df['Ticker'].unique():
            ticker_data = self.df[self.df['Ticker'] == ticker]
            plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker, linewidth=2)
        plt.title('Close Price Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        # add legend
        plt.legend()
        plt.xticks(rotation=45)
        # displaye the plot
        plt.tight_layout()
        plt.show
    def daily_percentage_change(self):
        """
        Calculates and plots the daily percentage change for each ticker to observe volatility.
        """
        self.df['Daily Percentage Change'] = self.df.groupby('Ticker')['Close'].pct_change() * 100
        
        plt.figure(figsize=(10, 6))
        
        for ticker in self.df['Ticker'].unique():
            ticker_data = self.df[self.df['Ticker'] == ticker]
            plt.plot(ticker_data['Date'], ticker_data['Daily Percentage Change'], label=ticker, linewidth=2)
        
        plt.title('Daily Percentage Change (Volatility) Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Percentage Change (%)', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def rolling_analysis(self):
        """
        Calculate and plot the rolling mean and rolling standard deviation to understand short-term trends and fluctuations.
        """
        plt.figure(figsize=(14, 8))
        
        for ticker in self.df['Ticker'].unique():
            ticker_data = self.df[self.df['Ticker'] == ticker]
            
            # Calculate rolling mean and rolling std
            ticker_data['Rolling Mean'] = ticker_data['Close'].rolling(window=self.window).mean()
            ticker_data['Rolling Std'] = ticker_data['Close'].rolling(window=self.window).std()
            
            # Plot the close price, rolling mean, and rolling standard deviation
            plt.plot(ticker_data['Date'], ticker_data['Close'], label=f'{ticker} Close', linewidth=1)
            plt.plot(ticker_data['Date'], ticker_data['Rolling Mean'], label=f'{ticker} Rolling Mean ({self.window} days)', linewidth=2, linestyle='--')
            plt.plot(ticker_data['Date'], ticker_data['Rolling Std'], label=f'{ticker} Rolling Std ({self.window} days)', linewidth=2, linestyle=':')

        plt.title(f'Rolling Mean and Standard Deviation of Close Price Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def outlier_detection_and_analysis(self, threshold=3):
        # Calculate daily returns for Close and Low columns
        self.df['Daily Return Close'] = self.df.groupby('Ticker')['Close'].pct_change() * 100
        self.df['Daily Return Low'] = self.df.groupby('Ticker')['Low'].pct_change() * 100
        
        # Use transform to calculate z-scores for both 'Close' and 'Low' returns
        self.df['Z-Score Close'] = self.df.groupby('Ticker')['Daily Return Close'].transform(zscore)
        self.df['Z-Score Low'] = self.df.groupby('Ticker')['Daily Return Low'].transform(zscore)
        
        # Identify outliers for both 'Close' and 'Low' returns (z-score > threshold or < -threshold)
        self.df['Outlier Close'] = self.df['Z-Score Close'].apply(lambda x: 'Outlier' if abs(x) > threshold else 'Normal')
        self.df['Outlier Low'] = self.df['Z-Score Low'].apply(lambda x: 'Outlier' if abs(x) > threshold else 'Normal')
        
        # Print the number of anomalies for each ticker
        outlier_counts = self.df.groupby('Ticker').apply(lambda x: (x['Outlier Close'] == 'Outlier').sum() + (x['Outlier Low'] == 'Outlier').sum())
        print("Number of outliers per ticker:")
        print(outlier_counts)
        
        # Plot the anomalies (optional)
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Plot anomalies for 'Close' and 'Low' returns
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.df, x='Date', y='Daily Return Close', hue='Outlier Close', palette='coolwarm', s=100, edgecolor='black')
        plt.title('Anomalies in Daily Return Close')
        plt.xlabel('Date')
        plt.ylabel('Daily Return Close')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.df, x='Date', y='Daily Return Low', hue='Outlier Low', palette='coolwarm', s=100, edgecolor='black')
        plt.title('Anomalies in Daily Return Low')
        plt.xlabel('Date')
        plt.ylabel('Daily Return Low')
        plt.show()
    def process_and_visualize_outliers(self):
        # Step 1: Calculate IQR for 'Daily Return Close' and detect outliers
        Q1_close = self.df['Daily Return Close'].quantile(0.25)
        Q3_close = self.df['Daily Return Close'].quantile(0.75)
        IQR_close = Q3_close - Q1_close
        lower_bound_close = Q1_close - 1.5 * IQR_close
        upper_bound_close = Q3_close + 1.5 * IQR_close
        self.df['Outlier Close'] = ((self.df['Daily Return Close'] < lower_bound_close) | (self.df['Daily Return Close'] > upper_bound_close))

        # Step 2: Calculate IQR for 'Daily Return Low' and detect outliers
        Q1_low = self.df['Daily Return Low'].quantile(0.25)
        Q3_low = self.df['Daily Return Low'].quantile(0.75)
        IQR_low = Q3_low - Q1_low
        lower_bound_low = Q1_low - 1.5 * IQR_low
        upper_bound_low = Q3_low + 1.5 * IQR_low
        self.df['Outlier Low'] = ((self.df['Daily Return Low'] < lower_bound_low) | (self.df['Daily Return Low'] > upper_bound_low))

        # Step 3: Calculate outlier percentages for each ticker for both columns
        outlier_percentage_close = self.df.groupby('Ticker')['Outlier Close'].mean() * 100
        outlier_percentage_low = self.df.groupby('Ticker')['Outlier Low'].mean() * 100
        
        # Display outlier percentages for each ticker
        print("Outlier Percentage for 'Daily Return Close' by Ticker:")
        print(outlier_percentage_close)
        print("\nOutlier Percentage for 'Daily Return Low' by Ticker:")
        print(outlier_percentage_low)

        # Step 4: Generate Box Plot for 'Daily Return Close' for each ticker
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='Ticker', y='Daily Return Close', hue='Outlier Close', palette='coolwarm')
        plt.title('Box Plot of Daily Return Close for Each Ticker')
        plt.xlabel('Ticker')
        plt.ylabel('Daily Return Close')
        plt.legend(title='Outlier Status')
        
        # Adding outlier percentages on the plot
        for i, ticker in enumerate(self.df['Ticker'].unique()):
            plt.text(i, self.df['Daily Return Close'].max(), f"{outlier_percentage_close[ticker]:.2f}%", 
                    ha='center', va='bottom', fontsize=10, color='black')
        
        plt.show()

        # Step 5: Generate Box Plot for 'Daily Return Low' for each ticker
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='Ticker', y='Daily Return Low', hue='Outlier Low', palette='coolwarm')
        plt.title('Box Plot of Daily Return Low for Each Ticker')
        plt.xlabel('Ticker')
        plt.ylabel('Daily Return Low')
        plt.legend(title='Outlier Status')
        
        # Adding outlier percentages on the plot
        for i, ticker in enumerate(self.df['Ticker'].unique()):
            plt.text(i, self.df['Daily Return Low'].max(), f"{outlier_percentage_low[ticker]:.2f}%", 
                    ha='center', va='bottom', fontsize=10, color='black')
        
        plt.show()

    
    def decompose_time_series(self, period=365, model='additive'):
        """
        Decompose the time series for each ticker in the DataFrame into trend, seasonal, and residual components.

        Args:
            period (int): The number of periods in a full seasonal cycle (default is 365 for yearly seasonality).
            model (str): The type of decomposition model ('additive' or 'multiplicative').

        Returns:
            None: Displays decomposition plots for each ticker.
        """
        # Ensure 'Date' is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        
        # Iterate over each unique ticker in the DataFrame
        tickers = self.df['Ticker'].unique()

        for ticker in tickers:
            ticker_data = self.df[self.df['Ticker'] == ticker]
            
            # Decompose the time series
            result = seasonal_decompose(ticker_data['Close'], model=model, period=period)
            
            # Plot the decomposition
            plt.figure(figsize=(12, 8))
            
            plt.subplot(411)
            plt.plot(result.observed)
            plt.title(f'Observed for {ticker}')
            
            plt.subplot(412)
            plt.plot(result.trend)
            plt.title(f'Trend for {ticker}')
            
            plt.subplot(413)
            plt.plot(result.seasonal)
            plt.title(f'Seasonal for {ticker}')
            
            plt.subplot(414)
            plt.plot(result.resid)
            plt.title(f'Residual for {ticker}')
            
            plt.tight_layout()
            plt.show()

    def analyze_volatility(self, window=30):
        """
        Analyze volatility for each ticker using rolling means and standard deviations.
        
        Args:
            window (int): The window size for calculating rolling means and standard deviations (default is 30).

        Returns:
            None: Displays rolling mean and standard deviation plots for each ticker.
        """
    
        # Iterate over each unique ticker in the DataFrame
        tickers = self.df['Ticker'].unique()
        
        for ticker in tickers:
            ticker_data = self.df[self.df['Ticker'] == ticker]

            # Calculate rolling mean and rolling standard deviation
            ticker_data['Rolling Mean'] = ticker_data['Close'].rolling(window=window).mean()
            ticker_data['Rolling Std'] = ticker_data['Close'].rolling(window=window).std()

            # Plot the rolling mean and standard deviation
            plt.figure(figsize=(14, 7))
            
            # Plot the closing price, rolling mean, and rolling standard deviation
            plt.plot(ticker_data['Close'], label=f'Close Price for {ticker}', color='blue', alpha=0.6)
            plt.plot(ticker_data['Rolling Mean'], label=f'{window}-Day Rolling Mean', color='red', linestyle='--')
            plt.plot(ticker_data['Rolling Std'], label=f'{window}-Day Rolling Std', color='green', linestyle='--')
            
            plt.title(f'Volatility Analysis for {ticker}')
            plt.legend(loc='best')
            plt.show()
    def calculate_daily_returns(self):
        """
        Calculates the daily percentage change for 'Close' price for each ticker and adds it to the DataFrame.
        """
        # Calculate daily returns (percentage change) for each ticker
        self.df['Daily Returns'] = self.df.groupby('Ticker')['Close'].pct_change() * 100
        
        # Optionally, calculate the daily returns for other columns (e.g., 'Low', 'Open', etc.)
        self.df['Daily Returns Low'] = self.df.groupby('Ticker')['Low'].pct_change() * 100
        
        return self.df
    
    def calculate_VaR(self, confidence_level=0.95, window=30):
        """
        Calculate the Value at Risk (VaR) for the stock using historical simulation.
        Args:
            confidence_level (float): The confidence level (default 0.95).
            window (int): The rolling window for calculating VaR (default 30 days).
        """
        # Ensure daily returns are calculated before proceeding
        if 'Daily Returns' not in self.df.columns:
            self.df = self.calculate_daily_returns()
        
        # Calculate rolling VaR at the specified confidence level
        rolling_VaR = self.df['Daily Returns'].rolling(window=window).apply(
            lambda x: np.percentile(x, (1 - confidence_level) * 100)
        )
        
        # Plot the rolling VaR
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Close'], label='Close Price', alpha=0.6)
        plt.plot(rolling_VaR, label=f'Rolling {window}-Day VaR at {confidence_level*100}% Confidence', color='orange')
        plt.title(f'Value at Risk (VaR) Analysis for Tesla Stock')
        plt.legend(loc='best')
        plt.show()
        
        return rolling_VaR
    
    def calculate_Sharpe_Ratio(self, risk_free_rate=0.02):
        """
        Calculate the Sharpe Ratio for the stock based on daily returns.
        """
        # Ensure daily returns are calculated before proceeding
        if 'Daily Returns' not in self.df.columns:
            self.df = self.calculate_daily_returns()
        
        # Calculate excess returns
        excess_returns = self.df['Daily Returns'] - risk_free_rate / 252  # Daily risk-free rate (approx. 252 trading days)
        
        # Calculate average and standard deviation of excess returns
        avg_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()
        
        # Calculate the Sharpe Ratio
        sharpe_ratio = avg_excess_return / std_excess_return
        return sharpe_ratio

    def document_key_insights(self):
        """
        Generate insights like overall direction of Tesla’s stock price, 
        fluctuations in daily returns, VaR and Sharpe Ratio to assess potential 
        losses and risk-adjusted returns.
        """
        # Ensure daily returns are calculated before proceeding
        if 'Daily Returns' not in self.df.columns:
            self.df = self.calculate_daily_returns()

        # Overall direction of Tesla's stock price
        cumulative_return = (1 + self.df['Daily Returns'] / 100).cumprod() - 1
        overall_direction = cumulative_return.iloc[-1] * 100  # Cumulative return at the end in percentage

        print(f"Overall Direction of Tesla's Stock Price: {overall_direction:.2f}%")

        # Fluctuations in daily returns
        daily_return_fluctuations = self.df['Daily Returns'].describe()  # Summary statistics
        print("\nFluctuations in Daily Returns:")
        print(daily_return_fluctuations)

        # Calculate and print VaR and Sharpe Ratio
        var = self.calculate_VaR()
        sharpe_ratio = self.calculate_Sharpe_Ratio()

        print(f"\nValue at Risk (VaR): {var.iloc[-1]:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")