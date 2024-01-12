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

    # Save the last date of the actual data to mark on the chart
    last_actual_date = data.index[-1]

    # Select the model and generate forecast
    if model_choice == 'Moving Average':
        forecast = moving_average(data, window)
    elif model_choice == 'Exponential Smoothing':
        forecast = exponential_smoothing(data, alpha)
    elif model_choice == 'Holt-Winters':
        forecast = holt_winters(data, period)

    # Create a combined DataFrame with actual and forecasted data
    combined = pd.DataFrame({
        'Actual Close': data,
        'Forecast': forecast
    })

    # Plot the actual and forecasted data using Streamlit's line_chart
    st.line_chart(combined)

    # Annotate last actual data point
    st.markdown(f"**Last Actual Close Price Date:** {last_actual_date.strftime('%Y-%m-%d')}")

# The code for plotting using matplotlib has been removed as per your request.
