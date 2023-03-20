import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from prophet import Prophet
import plotly
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

def get_model(ticker):
    # Use yf.Ticker to get historical data
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period='max', interval='1d')[['Close']]
    data = data.rename(columns={'Close': 'y'})
    data.columns = ['y']
    data['ds'] = data.index
    data['y'] = data['y']/100

    # Convert timezone-aware column to timezone-naive column
    data['ds'] = data['ds'].dt.tz_localize(None)

    # Fit a Prophet model to the data
    model = Prophet(interval_width=0.95,
                    changepoint_prior_scale=0.5,
                    seasonality_prior_scale=0.01
                    #seasonality_mode='additive'
                    )
    model.fit(data)

    return model


def run_prophet(model, start_date, end_date):
    # Make future dataframe for prediction date range
    future = model.make_future_dataframe(periods=(end_date - start_date).days + 1, include_history=False)

    # Make prediction
    forecast = model.predict(future)

    # Filter prediction for selected date range
    forecast = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]

    # Merge with original data to get the actual values
    data = model.history[['ds', 'y']]
    data = data[(data['ds'] >= pd.to_datetime(start_date)) & (data['ds'] <= pd.to_datetime(end_date))]
    forecast = pd.merge(forecast, data, on='ds', how='left')
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

