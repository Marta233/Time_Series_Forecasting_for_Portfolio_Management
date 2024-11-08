import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import os

class FinancialDataAnalysis:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def download_data(self):
        """
        Downloads historical data for the specified tickers and date range.
        """
        for ticker in self.tickers:
            self.data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)
        print("Data downloaded successfully!")

    def save_to_files(self, directory="financial_data"):
        """
        Saves each ticker's data as a separate CSV file in the specified directory,
        cleaning up the first few rows (if necessary) and saving the cleaned data.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        for ticker, df in self.data.items():
            # Reset the index so 'Date' becomes a column
            df.reset_index(inplace=True)  # Ensure 'Date' is a column, not the index
            
            # Save cleaned data to a CSV file
            df.to_csv(f"{directory}/{ticker}.csv", index=False)

        print(f"Data saved to directory: {directory}")
    
    def save_to_merged_file(self, directory="financial_data"):
        """
        Merges all tickers' data into one DataFrame and saves it as a single CSV file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        merged_data = pd.DataFrame()  # Create an empty DataFrame to store the merged data

        # Loop through each ticker's data and add the 'Ticker' column
        for ticker, df in self.data.items():
            df.reset_index(inplace=True)  # Ensure 'Date' is a column, not the index
            df['Ticker'] = ticker  # Add a new column for the ticker name
            # Ensure columns are correctly named and structured
            df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume','Ticker']
            merged_data = pd.concat([merged_data, df], ignore_index=True)  # Merge the data
            
        # Save the merged data to a CSV file
        merged_data.to_csv(f"{directory}/merged_financial_data.csv", index=False)
        print(f"Data merged and saved to {directory}/merged_financial_data.csv")
