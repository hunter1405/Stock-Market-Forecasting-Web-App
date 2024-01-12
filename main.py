import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Example function to load data and create a forecast
def forecast_stock(stock_code, period='M'):
    # Load the data - this would be replaced with actual data loading logic
    # For example, using pandas_datareader: df = pdr.get_data_yahoo(stock_code)
    df = pd.read_csv('path_to_your_stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Assume 'Close' is the column we want to forecast
    series = df['Close']

    # Resample the series according to the selected period (daily, weekly, monthly)
    if period == 'D':  # Daily
        resampled_series = series.asfreq('D')
    elif period == 'W':  # Weekly
        resampled_series = series.resample('W').mean()
    elif period == 'M':  # Monthly
        resampled_series = series.resample('M').mean()

    # Fit an ARIMA model - parameters (p, d, q) would need to be optimized for each stock
    model = ARIMA(resampled_series, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast the next period
    forecast = model_fit.forecast(steps=1)[0]

    return forecast

# Example usage
stock_code = 'AAPL'  # This would come from the user input
forecast_period = 'M'  # This would also come from the user input, 'D', 'W', or 'M'

forecast = forecast_stock(stock_code, forecast_period)

# This forecast value would then be sent back to the frontend to display in a chart
print(f"The forecast for stock {stock_code} for the next {forecast_period} is: {forecast}")
