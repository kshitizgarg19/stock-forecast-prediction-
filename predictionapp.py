import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast Prediction')

markets = ['US', 'India']
market = st.radio('Select market:', markets)

if market == 'US':
    options = {
        'Apple (AAPL)': 'AAPL',
        'Microsoft (MSFT)': 'MSFT',
        'Amazon (AMZN)': 'AMZN',
        'Google (GOOGL)': 'GOOGL',
        'Tesla (TSLA)': 'TSLA',
        'S&P 500 ETF (SPY)': 'SPY',  # Example of US index fund
    }
    currency = '$'
else:
    options = {
        'Infosys (INFY)': 'INFY',
        'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
        'Tata Consultancy Services (TCS.NS)': 'TCS.NS',
        'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
        'ICICI Bank (ICICIBANK.NS)': 'ICICIBANK.NS'
          # Example of Indian index fund (Bank Nifty)
    }
    currency = '₹'

selected_stock = st.selectbox('Select dataset for prediction', list(options.keys()))
ticker = options[selected_stock]

custom_ticker = st.text_input('Or enter a custom ticker (optional):')
if custom_ticker:
    ticker = custom_ticker

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Check if df_train has valid data
if len(data) == 0:
    st.error('No data available for the selected stock. Please choose another stock or check your custom ticker.')
else:
    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Prepare the data for linear regression
    df_train = data[['Date', 'Close']]
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Date_ordinal'] = df_train['Date'].map(pd.Timestamp.toordinal)  # Fix: Removed parentheses

    if len(df_train) == 0:
        st.error('No valid data available for the selected stock. Please choose another stock or check your custom ticker.')
    else:
        X = df_train[['Date_ordinal']]
        y = df_train['Close']

        # Scaling the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit the model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predict the next day
        next_day = (pd.to_datetime(TODAY) + timedelta(days=1)).to_pydatetime()
        next_day_ordinal = np.array([[next_day.toordinal()]])
        next_day_scaled = scaler.transform(next_day_ordinal)

        # Make prediction
        next_day_prediction = model.predict(next_day_scaled)

        # Create a DataFrame for the forecast
        forecast = pd.DataFrame({'Date': [next_day], 'Predicted Close': next_day_prediction})

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast)

        # Plot the forecast data
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], name="Actual Close"))
        fig1.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], name="Predicted Close", line=dict(color='royalblue')))
        fig1.layout.update(title_text='Forecast plot', xaxis_rangeslider_visible=True, xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig1)

        # Components plot (simple plot of the linear regression line)
        st.write("Forecast components")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], mode='markers', name='Actual'))
        fig2.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted Close'], mode='lines', name='Forecast', line=dict(color='royalblue')))
        fig2.layout.update(title_text='Forecast components', xaxis_title='Date', yaxis_title=f'Price ({currency})')
        st.plotly_chart(fig2)

        # Display the predicted prices in a box
        latest_prediction = forecast.iloc[-1]
        st.subheader('Predicted Price')
        st.markdown(f"<div style='text-align: center; font-size: 24px;'>Predicted Close Price on {latest_prediction['Date'].strftime('%d-%m-%Y')}: {currency}{latest_prediction['Predicted Close']:.2f}</div>", unsafe_allow_html=True)

# Developer attribution
st.markdown("""
    <br><br><br>
    <div style="text-align: center;">
        <p>Developed and maintained by Kshitiz Garg</p>
    </div>
""", unsafe_allow_html=True)
