import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from transformers import pipeline 

def calculate_brokerage(amount):
    return min(20, 0.0005 * amount)

def fetch_live_data(stock_symbol, interval='5m', period='1mo'):
    live_data = yf.download(tickers=stock_symbol, period=period, interval=interval)
    return live_data

def calculate_signals(data, short_window=7, long_window=30):
    data['Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['Signal'] = np.where(data['Short'] > data['Long'], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()
    return data

def simulate_portfolio(data, initial_capital=100000):
    # Initialize portfolio DataFrame with matching indices to data
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = initial_capital
    portfolio['Total'] = initial_capital
    portfolio['Returns'] = 0.0
    shares_held = 0

    for i in range(1, len(data)):
        # Ensure that the current row exists in portfolio before assignment
        current_index = data.index[i]

        # Buy Signal
        if data['Position'].iloc[i] == 1.0:
            shares_to_buy = portfolio['Cash'].iloc[i - 1] // data['Close'].iloc[i]
            amount = shares_to_buy * data['Close'].iloc[i]
            brokerage = calculate_brokerage(amount)
            tcost = amount + brokerage
            if tcost <= portfolio['Cash'].iloc[i - 1]:
                shares_held += shares_to_buy
                portfolio.at[current_index, 'Cash'] = portfolio['Cash'].iloc[i - 1] - tcost
            else:
                portfolio.at[current_index, 'Cash'] = portfolio['Cash'].iloc[i - 1]
        
        # Sell Signal
        elif data['Position'].iloc[i] == -1.0 and shares_held > 0:
            amount = shares_held * data['Close'].iloc[i]
            brokerage = calculate_brokerage(amount)
            tcost = amount - brokerage
            portfolio.at[current_index, 'Cash'] = portfolio['Cash'].iloc[i - 1] + tcost
            shares_held = 0
        else:
            portfolio.at[current_index, 'Cash'] = portfolio['Cash'].iloc[i - 1]

        # Update holdings, total, and returns using .at[]
        portfolio.at[current_index, 'Holdings'] = shares_held * data['Close'].iloc[i]
        portfolio.at[current_index, 'Total'] = portfolio.at[current_index, 'Cash'] + portfolio.at[current_index, 'Holdings']
        portfolio.at[current_index, 'Returns'] = portfolio.at[current_index, 'Total'] - portfolio['Total'].iloc[i - 1]

    return portfolio


def plot_stock_data(data, portfolio, timeframe):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Stock Price')
    plt.plot(data.index, data['Short'], label='Short Moving Average (7-period)')
    plt.plot(data.index, data['Long'], label='Long Moving Average (30-period)')

    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    
    plt.plot(buy_signals.index, data['Close'][buy_signals.index], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(sell_signals.index, data['Close'][sell_signals.index], 'v', markersize=10, color='r', label='Sell Signal')

    plt.title(f'Stock Price with Moving Averages ({timeframe})')
    plt.xlabel('Time')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot(plt)

def plot_profit_chart(portfolio, timeframe):
    plt.figure(figsize=(14, 5))
    plt.plot(portfolio.index, portfolio['Total'], label='Total Portfolio Value')
    plt.title(f'Portfolio Value Over Time ({timeframe})')
    plt.xlabel('Time')
    plt.ylabel('Total Value (INR)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot(plt)

def streamlit_interface():
    st.title('Live Stock Portfolio Simulation')
    
    st.subheader('Enter the stock name and relax, let me suggest when to buy and sell to maximize profit')
    stock_symbol = st.text_input("Enter Stock Name (e.g., TATAMOTORS.NS) (note: must add .NS after the stock name)", "TATAMOTORS.NS")
    initial_capital = st.number_input("Enter Initial Capital (INR)", min_value=1000, value=10000)

    timeframe_option = st.selectbox("Select Timeframe for Data", ["1 Day", "1 month"])

    if st.button("Run Simulation"):
        st.write("Fetching live data and running simulation...")
        period = '1mo'
        interval = '5m'

        if timeframe_option == "1 Day":
            period = '1d'
            interval = '1m'
        
        data = fetch_live_data(stock_symbol, interval=interval, period=period)
        data = calculate_signals(data)
        st.write("Fetched Data:", data)
        if data.empty:
            st.error("No data fetched. Please check the stock symbol or try a different timeframe.")
        else:
            portfolio = simulate_portfolio(data, initial_capital)

            st.write(f"Latest data for {stock_symbol}:")
            st.dataframe(data.tail())
            
            st.write("Portfolio summary:")
            st.dataframe(portfolio.tail())

            st.write(f"Stock Price and Moving Averages ({timeframe_option}):")
            plot_stock_data(data, portfolio, timeframe_option)
            
            st.write(f"Portfolio Value Over Time ({timeframe_option}):")
            plot_profit_chart(portfolio, timeframe_option)

if __name__ == "__main__":
    streamlit_interface()
