import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pmdarima import auto_arima

class TimeSeriesProcessor:
    def __init__(self, file_path, date_column, target_column):
        """
        Initialize the time series processor with the data file path.
        :param file_path: Path to the CSV file containing the data.
        :param date_column: Name of the column containing dates.
        :param target_column: Name of the column containing target values for ARIMA.
        """
        self.file_path = file_path
        self.date_column = date_column
        self.target_column = target_column
        self.data = None
        self.time_series = None
        self.model = None
        self.model_fit = None
        self.differenced_data = None
        self.lstm_model = None

    def load_and_prepare_data(self):
        """ Load the data, parse dates, and set up the target time series. """
        self.data = pd.read_csv(self.file_path)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.data.set_index(self.date_column, inplace=True)
        self.time_series = self.data[self.target_column].dropna()

    def plot_time_series(self):
        """ Plot the original time series data. """
        self.time_series.plot(title="Time Series Data", figsize=(10, 5))
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.show()

    def check_stationarity(self):
        """ Perform the Augmented Dickey-Fuller test to check for stationarity. """
        result = adfuller(self.time_series)
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        return result[1]

    def make_stationary(self):
        """ Apply differencing to make the data stationary. """
        self.differenced_data = self.time_series.diff().dropna()
        return self.differenced_data

    def plot_acf_pacf(self):
        """ Plot ACF and PACF to determine ARIMA parameters. """
        if self.differenced_data is None:
            print("Differencing the data first for stationarity.")
            self.make_stationary()
        plot_acf(self.differenced_data, lags=20)
        plot_pacf(self.differenced_data, lags=20)
        plt.show()

    def fit_arima(self, p, d, q):
        """ Fit the ARIMA model with given parameters. """
        self.model = ARIMA(self.time_series, order=(p, d, q))
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())

    def fit_sarima(self, p, d, q, P, D, Q, s):
        """ Fit the SARIMA model with given parameters. """
        self.model = SARIMAX(self.time_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())

    def optimize_arima(self):
        """ Use auto_arima to find the best (p, d, q) parameters. """
        model = auto_arima(self.time_series, seasonal=False, stepwise=True, trace=True)
        print(model.summary())
        return model.order

    def optimize_sarima(self, seasonal_period):
        """ Use auto_arima to find the best SARIMA parameters. """
        model = auto_arima(self.time_series, seasonal=True, m=seasonal_period, stepwise=True, trace=True)
        print(model.summary())
        return model.order, model.seasonal_order

    def forecast(self, steps=30):
        """ Forecast future values using the fitted model. """
        if not self.model_fit:
            raise ValueError("Model is not fitted. Call fit_model() first.")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def plot_forecast(self, steps=30, test_data=None, arima_forecast=None, sarima_forecast=None, lstm_forecast=None):
        """ Plot actual data and forecasts. """
        plt.figure(figsize=(10, 5))
        
        # Plot actual test data
        if test_data is not None:
            plt.plot(test_data.index[-steps:], test_data.values[-steps:], label="Actual Data", color='blue')
        
        # Plot ARIMA forecast
        if arima_forecast is not None:
            plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", color='red')
        
        # Plot SARIMA forecast
        if sarima_forecast is not None:
            plt.plot(sarima_forecast.index, sarima_forecast, label="SARIMA Forecast", color='green')
        
        # Plot LSTM forecast
        if lstm_forecast is not None:
            plt.plot(test_data.index[-steps:], lstm_forecast, label="LSTM Forecast", color='orange')
        
        plt.legend()
        plt.show()


    def evaluate_model(self, test_data, model_name):
        """ Evaluate the model using Mean Absolute Error and Root Mean Squared Error. """
        if not self.model_fit:
            raise ValueError("Model is not fitted. Call fit_model() first.")
        
        steps = len(test_data)
        forecast = self.forecast(steps=steps)
        
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        print(f"{model_name} - Mean Absolute Error (MAE): {mae}")
        print(f"{model_name} - Root Mean Squared Error (RMSE): {rmse}")
        return mae, rmse

    def fit_lstm(self, data, n_steps, epochs=200):
        """
        Train the LSTM model.
        :param data: Input time series as a 1D NumPy array.
        :param n_steps: Number of time steps for the input sequence.
        :param epochs: Number of epochs for training.
        """
        # Prepare data
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        X, y = np.array(X), np.array(y)

        # Reshape to [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Add a single feature dimension

        # Define LSTM model
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
        self.lstm_model.add(Dense(1))  # Output layer
        self.lstm_model.compile(optimizer='adam', loss='mse')

        # Fit the model
        self.lstm_model.fit(X, y, epochs=epochs, verbose=0)

    def predict_lstm(self, data, n_steps):
        """
        Perform a rolling forecast using the trained LSTM model.
        :param data: The input time series as a 1D NumPy array.
        :param n_steps: Number of time steps for the input sequence.
        :return: List of forecasted values.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been trained. Call the 'fit_lstm' method first.")

        # Prepare for rolling forecast
        predictions = []
        input_data = data[-n_steps:].tolist()

        for _ in range(len(data)):
            input_array = np.array(input_data[-n_steps:]).reshape((1, n_steps, 1))
            prediction = self.lstm_model.predict(input_array, verbose=0)[0][0]
            predictions.append(prediction)
            input_data.append(prediction)  # Append prediction for the next step

        return predictions

