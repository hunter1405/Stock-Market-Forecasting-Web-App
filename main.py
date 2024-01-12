import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas_datareader.data as web
from datetime import datetime

# Function to fetch stock data using pandas_datareader
def fetch_stock_data(stock_code, start_date, end_date, data_type='Close'):
    try:
        stock_data = web.DataReader(stock_code, 'yahoo', start_date, end_date)
        print(stock_data)  # Debug: Print the raw data
        return stock_data[data_type]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Functions for forecasting models
def moving_average(series, window):
    return series.rolling(window).mean()

def exponential_smoothing(series, alpha):
    return series.ewm(alpha=alpha).mean()

def holt_winters(series, period):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=period)
    model_fit = model.fit()
    return model_fit.fittedvalues

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
if st.button('Generate Forecast', key='generate_forecast'):
    # Use input from the data fetching UI if no new ticker is provided
    forecast_stock_code = forecast_stock_code or input_stock_code
    if forecast_stock_code:
        with st.spinner(f"Generating forecast for {forecast_stock_code}..."):
            # Fetch the stock data for forecasting
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
