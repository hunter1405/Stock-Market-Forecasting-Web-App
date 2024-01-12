import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf

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

# Data fetching UI
st.subheader('Fetch Stock Data')
input_stock_code = st.text_input('Enter Stock Ticker', 'AAPL').upper()
selected_data_type = st.selectbox('Select Data Type', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
start_date = st.date_input('Start Date', pd.to_datetime('2021-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

if st.button('Fetch Data'):
    with st.spinner(f"Fetching data for {input_stock_code}..."):
        fetched_data = yf.download(input_stock_code, start=start_date, end=end_date)[selected_data_type]
        st.dataframe(fetched_data)

        # Download button for the fetched data
        csv = fetched_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'{input_stock_code}_{selected_data_type}.csv',
            mime='text/csv',
        )

# Forecasting UI
st.subheader('Forecasting')
forecast_stock_code = st.text_input('Enter Stock Code for Forecast', 'AAPL')

# Create empty containers for user input to prevent rerun errors
window = None
alpha = None
period = None
forecast = pd.Series()

# Use the input from the data fetching UI if the user has not provided a new ticker
if not forecast_stock_code:
    forecast_stock_code = input_stock_code

model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

if model_choice == 'Moving Average':
    window = st.slider('Moving Average Window', 3, 30, 3)
elif model_choice == 'Exponential Smoothing':
    alpha = st.slider('Alpha', 0.01, 1.0, 0.1)
elif model_choice == 'Holt-Winters':
    period = st.slider('Seasonal Period', 2, 12, 4)

if st.button('Generate Forecast'):
    with st.spinner(f"Generating forecast for {forecast_stock_code}..."):
        # Fetch the stock data for forecasting
        data = yf.download(forecast_stock_code, start=start_date, end=end_date)['Close']

        # Select the model and generate
