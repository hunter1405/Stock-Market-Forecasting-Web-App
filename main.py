import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import matplotlib.pyplot as plt

# Function to fetch stock data using yfinance
def fetch_stock_data(stock_code, period='1d'):
    stock_data = yf.Ticker(stock_code)
    return stock_data.history(period=period)

# Define the Moving Average function
def moving_average(data, window):
    return data['Close'].rolling(window).mean()

# Define the Exponential Smoothing function
def exponential_smoothing(data, alpha):
    model = SimpleExpSmoothing(data['Close'])
    model_fit = model.fit(smoothing_level=alpha)
    return model_fit.fittedvalues

# Define the Holt-Winters function
def holt_winters(data, period):
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=period)
    model_fit = model.fit()
    return model_fit.fittedvalues

# Streamlit UI
st.title('Stock Forecasting Application')

# Input for stock ticker symbol
stock_code = st.text_input('Enter Stock Ticker Symbol', 'AAPL')

# Dropdown for choosing the forecasting model
model_choice = st.selectbox('Choose the Forecasting Model', ['Moving Average', 'Exponential Smoothing', 'Holt-Winters'])

