pip install yfinance
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.dates as mdates
import yfinance as yf

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

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(stock_code, start_date, end_date, data_type):
    stock_data = yf.download(stock_code, start=start_date, end=end_date)
    return stock_data[data_type]

# Streamlit UI
st.title('Stock Forecasting Application')

# Data fetching UI
st.subheader('Fetch Stock Data')
input_stock_code = st.text_input('Enter Stock Ticker', 'AAPL').upper()
selected_data_type = st.selectbox('Select Data Type', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
start_date = st.date_input('Start Date', pd.to_datetime('2021-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

if st.button('Fetch Data'):
    fetched_data = fetch_stock_data(input_stock_code, start_date, end_date, selected_data_type)
    st.dataframe(fetched_data)

    # Download button for the fetched data
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df_to_csv(fetched_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'{input_stock_code}_{selected_data_type}.csv',
        mime='text/csv',
    )

# Forecasting UI
st.subheader('Forecasting')
stock_code = st.text_input('Enter Stock Code for Forecast', 'AAPL')

model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

if model_choice == 'Moving Average':
    window = st.slider('Moving Average Window', 3, 30, 3)
elif model_choice == 'Exponential Smoothing':
    alpha = st.slider('Alpha', 0.01, 1.0, 0.1)
elif model_choice == 'Holt-Winters':
    period = st.slider('Seasonal Period', 2, 12, 4)

if st.button('Forecast Data'):
    # Fetch the stock data for forecasting
    data = fetch_stock_data(stock_code, '2020-01-01', '2021-01-01', 'Close')

    # Select the model and generate forecast
    if model_choice == 'Moving Average':
        forecast = moving_average(data, window)
    elif model_choice == 'Exponential Smoothing':
        forecast = exponential_smoothing(data, alpha)
    elif model_choice == 'Holt-Winters':
        forecast = holt_winters(data, period)

    # Display forecast
    st.line_chart(forecast)

    # Annotate last actual data point
    st.markdown(f"**Last Actual Close Price Date:** {last_actual_date.strftime('%Y-%m-%d')}")

# The code for plotting using matplotlib has been removed as per your request.
