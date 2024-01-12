import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf
from datetime import datetime

# Function to fetch stock data using yfinance
def fetch_stock_data(stock_code, start_date, end_date, data_type='Close'):
    stock_data = yf.download(stock_code, start=start_date, end=end_date)
    return stock_data[data_type]

# Streamlit UI for Data fetching
st.title('Stock Forecasting Application')

# Fetch Stock Data UI
st.subheader('Fetch Stock Data')
input_stock_code_fetch = st.text_input('Enter Stock Ticker to Fetch Data', 'AAPL', key='input_stock_code_fetch').upper()
data_type_selection = st.selectbox('Select Data Type', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], key='data_type_selection')
start_date = st.date_input('Start Date', datetime(2021, 1, 1), key='start_date')
end_date = st.date_input('End Date', datetime.today(), key='end_date')

# Fetch and display the data
if st.button('Fetch Data', key='fetch_data_button'):
    data = fetch_stock_data(input_stock_code_fetch, start_date, end_date, data_type_selection)
    st.write(data.head())  # Display the first few rows of the data
    csv = data.to_csv().encode('utf-8')
    st.download_button(label="Download data as CSV", data=csv, file_name=f'{input_stock_code_fetch}_{data_type_selection}.csv', mime='text/csv')

# Streamlit UI for Forecasting
st.subheader('Forecasting')
forecast_stock_code = st.text_input('Enter Stock Code for Forecast', key='forecast_stock_code')
model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'], key='model_choice')

# Depending on the model choice, display the appropriate widget to get the parameter(s)
window = alpha = period = None
if model_choice == 'Moving Average':
    window = st.slider('Moving Average Window', 3, 30, 3, key='window')
elif model_choice == 'Exponential Smoothing':
    alpha = st.slider('Alpha', 0.01, 1.0, 0.1, key='alpha')
elif model_choice == 'Holt-Winters':
    period = st.slider('Seasonal Period', 2, 12, 4, key='period')

# Generate forecast
if st.button('Generate Forecast', key='generate_forecast_button'):
    forecast_stock_code = forecast_stock_code or input_stock_code_fetch
    if forecast_stock_code:
        with st.spinner(f"Generating forecast for {forecast_stock_code}..."):
            forecast_data = fetch_stock_data(forecast_stock_code, start_date, end_date, data_type_selection)

            # Select the model and generate the forecast
            if model_choice == 'Moving Average':
                forecast_result = moving_average(forecast_data, window)
            elif model_choice == 'Exponential Smoothing':
                forecast_result = exponential_smoothing(forecast_data, alpha)
            elif model_choice == 'Holt-Winters':
                forecast_result = holt_winters(forecast_data, period)

            # Display the forecast
            st.line_chart(forecast_result)
    else:
        st.error('Please enter a stock code for forecasting.')
