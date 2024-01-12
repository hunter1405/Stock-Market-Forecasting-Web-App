import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas_datareader.data as web
from datetime import datetime

# Function to fetch stock data using pandas_datareader
def fetch_stock_data(stock_code, start_date, end_date, data_type='Close'):
    stock_data = web.DataReader(stock_code, 'yahoo', start_date, end_date)
    return stock_data[data_type]

# Streamlit UI for Data fetching
st.title('Stock Forecasting Application')
st.subheader('Fetch Stock Data')
input_stock_code = st.text_input('Enter Stock Ticker', 'AAPL', key='input_stock_code').upper()
data_type_selection = st.selectbox('Select Data Type', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], key='data_type_selection')
start_date = st.date_input('Start Date', datetime(2021, 1, 1), key='start_date')
end_date = st.date_input('End Date', datetime.today(), key='end_date')

# Fetch and display the data
if st.button('Fetch Data', key='fetch_data'):
    data = fetch_stock_data(input_stock_code, start_date, end_date, data_type_selection)
    st.write(data.head())  # Display the first few rows of the data
    csv = data.to_csv().encode('utf-8')
    st.download_button(label="Download data as CSV", data=csv, file_name=f'{input_stock_code}_{data_type_selection}.csv', mime='text/csv')

# Streamlit UI for Forecasting
st.subheader('Forecasting')
forecast
