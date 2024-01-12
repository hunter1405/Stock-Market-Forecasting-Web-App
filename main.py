import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.dates as mdates

# Mock function to simulate getting stock data
def get_stock_data(stock_code):
    rng = pd.date_range(pd.to_datetime('2021-01-01'), periods=100, freq='D')
    data = pd.DataFrame({'Date': rng, 'Close': np.random.rand(len(rng)) * 100})
    data.set_index('Date', inplace=True)
    return data['Close']

# Functions for forecasting models
def moving_average(series, window):
    return series.rolling(window).mean()

def exponential_smoothing(series, alpha):
    return series.ewm(alpha=alpha).mean()

def holt_winters(series, period):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=period)
    model_fit = model.fit()
    return model_fit.fittedvalues

# Streamlit UI
st.title('Stock Forecasting Application')

stock_code = st.text_input('Enter Stock Code', 'AAPL')

model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

window = None
alpha = None
period = None
forecast = None

if model_choice == 'Moving Average':
    window = st.slider('Moving Average Window', 3, 30, 3)
elif model_choice == 'Exponential Smoothing':
    alpha = st.slider('Alpha', 0.01, 1.0, 0.1)
elif model_choice == 'Holt-Winters':
    period = st.slider('Seasonal Period', 2, 12, 4)

if st.button('Forecast'):
    # Simulate fetching stock data
    data = get_stock_data(stock_code)

    # Select the model and generate forecast
    if model_choice == 'Moving Average':
        forecast = moving_average(data, window)
    elif model_choice == 'Exponential Smoothing':
        forecast = exponential_smoothing(data, alpha)
    elif model_choice == 'Holt-Winters':
        forecast = holt_winters(data, period)

    # Plotting the forecast data with formatted dates
    fig, ax = plt.subplots()
    ax.plot(forecast.index, forecast.values)

    # Set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Forecasted Stock Prices')

    # Render the plot in Streamlit
    st.pyplot(fig)

# Check if the forecast variable exists before attempting to plot
if forecast is not None:
    st.line_chart(forecast)
