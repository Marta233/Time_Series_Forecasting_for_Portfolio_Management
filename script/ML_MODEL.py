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

class TimeSeriesProcessor:
    def __init__(self, file_path, date_column, target_column):
        self.file_path = file_path
        self.date_column = date_column
        self.target_column = target_column
        self.data = None
        self.time_series = None
        self.model_fit = None
        self.differenced_data = None
        self.lstm_model = None
        self.scaler = MinMaxScaler()

    def load_and_prepare_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.data.set_index(self.date_column, inplace=True)
        self.time_series = self.data[self.target_column].dropna()

    def plot_time_series(self):
        self.time_series.plot(title="Time Series Data", figsize=(10, 5))
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.show()

    def check_stationarity(self):
        result = adfuller(self.time_series)
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        return result[1]

    def make_stationary(self):
        self.differenced_data = self.time_series.diff().dropna()
        return self.differenced_data

    def plot_acf_pacf(self):
        if self.differenced_data is None:
            self.make_stationary()
        plot_acf(self.differenced_data, lags=20)
        plot_pacf(self.differenced_data, lags=20)
        plt.show()

    def grid_search_arima(self, p_values, d_values, q_values):
        best_aic = float('inf')
        best_order = None
        best_model = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(self.time_series, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except Exception as e:
                        continue

        print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
        self.model_fit = best_model
        return best_model

    def grid_search_sarima(self, p_values, d_values, q_values, seasonal_p, seasonal_d, seasonal_q, s):
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        best_model = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for sp in seasonal_p:
                        for sd in seasonal_d:
                            for sq in seasonal_q:
                                try:
                                    seasonal_order = (sp, sd, sq, s)
                                    model = SARIMAX(self.time_series, order=(p, d, q), seasonal_order=seasonal_order)
                                    model_fit = model.fit(disp=False)  # Suppress output
                                    if model_fit.aic < best_aic:
                                        best_aic = model_fit.aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = seasonal_order
                                        best_model = model_fit
                                except Exception as e:
                                    print(f"Error fitting SARIMA({p},{d},{q})x({sp},{sd},{sq},{s}): {e}")
                                    continue

        if best_model is not None:
            print(f"Best SARIMA order: {best_order} with seasonal order: {best_seasonal_order} and AIC: {best_aic}")
            self.model_fit = best_model  # Ensure model_fit is set
            return best_model
        else:
            raise ValueError("No suitable SARIMA model found.")

    def forecast(self, steps=30):
        if self.model_fit is None:
            raise ValueError("Model is not fitted. Call fit_model() first.")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def evaluate_arima_sarima(self, test_data):
        if self.model_fit is None:
            raise ValueError("Model is not fitted. Fit the ARIMA/SARIMA model first.")

        steps = len(test_data)
        forecast = self.forecast(steps=steps)

        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        print(f"ARIMA/SARIMA - Mean Absolute Error (MAE): {mae}")
        print(f"ARIMA/SARIMA - Root Mean Squared Error (RMSE): {rmse}")

        return forecast

    def plot_results(self, train_data, test_data, forecast, model_name):
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Training Data', color='blue')
        plt.plot(test_data.index, test_data, label='Test Data', color='green')
        plt.plot(test_data.index, forecast, label='Forecast', color='red')
        plt.title(f'{model_name} Forecast vs Actual Data')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.show()

    def fit_lstm(self, train_data, n_steps=60, epochs=10, batch_size=32):
        # Scale training data
        train_scaled = self.scaler.fit_transform(np.array(train_data).reshape(-1, 1))

        # Prepare data for LSTM training
        X_train, y_train = [], []
        for i in range(n_steps, len(train_scaled)):
            X_train.append(train_scaled[i-n_steps:i])
            y_train.append(train_scaled[i])
        
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Define LSTM model architecture
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])

        # Compile and train the LSTM model
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_lstm(self, train_data, test_data, n_steps=60):
        # Prepare inputs for predictions
        inputs = self.scaler.transform(np.array(train_data[-n_steps:].tolist() + test_data.tolist()).reshape(-1, 1))
        X_test = []
        for i in range(n_steps, len(inputs)):
            X_test.append(inputs[i-n_steps:i])
        X_test = np.array(X_test)

        # Make predictions using the trained model
        lstm_forecast = self.lstm_model.predict(X_test)
        lstm_forecast = self.scaler.inverse_transform(lstm_forecast)  # Rescale predictions to original scale

        # Align the forecast with the test data
        return lstm_forecast[-len(test_data):]

    def evaluate_lstm(self, train_data, test_data):
        if self.lstm_model is None:
            raise ValueError("LSTM model is not fitted. Fit the LSTM model first.")

        lstm_forecast = self.predict_lstm(train_data, test_data)
        
        mae = mean_absolute_error(test_data, lstm_forecast)
        rmse = np.sqrt(mean_squared_error(test_data, lstm_forecast))
        print(f"LSTM - Mean Absolute Error (MAE): {mae}")
        print(f"LSTM - Root Mean Squared Error (RMSE): {rmse}")

        return lstm_forecast
    
    
