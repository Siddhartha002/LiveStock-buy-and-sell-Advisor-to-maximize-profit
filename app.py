import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from transformers import pipeline 
sentiment_analyzer = pipeline("sentiment-analysis")
def plot_sentiment_pie_chart(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title("Sentiment Analysis Results")
    return plt 

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
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = float(initial_capital)
    portfolio['Total'] = float(initial_capital)
    portfolio['Returns'] = 0.0
    shares_held = 0

    for i in range(1, len(data)):
        if data.iloc[i]['Position'] == 1:
            shares_to_buy = portfolio.iloc[i - 1]['Cash'] // data.iloc[i]['Close']
            amount = shares_to_buy * data.iloc[i]['Close']
            brokerage = calculate_brokerage(amount)
            tcost = amount + brokerage
            if tcost <= portfolio.iloc[i - 1]['Cash']:
                shares_held += shares_to_buy
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i - 1]['Cash'] - tcost
            else:
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i - 1]['Cash']
        elif data.iloc[i]['Position'] == -1 and shares_held > 0:
            amount = shares_held * data.iloc[i]['Close']
            brokerage = calculate_brokerage(amount)
            tcost = amount - brokerage
            portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i - 1]['Cash'] + tcost
            shares_held = 0
        else:
            portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio.iloc[i - 1]['Cash']

        portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = shares_held * data.iloc[i]['Close']
        portfolio.iloc[i, portfolio.columns.get_loc('Total')] = portfolio.iloc[i]['Cash'] + portfolio.iloc[i]['Holdings']
        portfolio.iloc[i, portfolio.columns.get_loc('Returns')] = portfolio.iloc[i]['Total'] - portfolio.iloc[i - 1]['Total']
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
    st.pyplot()

def plot_profit_chart(portfolio, timeframe):
    plt.figure(figsize=(14, 5))
    plt.plot(portfolio.index, portfolio['Total'], label='Total Portfolio Value')
    plt.title(f'Portfolio Value Over Time ({timeframe})')
    plt.xlabel('Time')
    plt.ylabel('Total Value (INR)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot()

def streamlit_interface():
    st.title('Live Stock Portfolio Simulation')
    user_input = st.text_area("Enter your comments or thoughts:")
    if st.button("Analyze Sentiment"):
        if user_input:
            result = sentiment_analyzer(user_input)
            sentiment = result[0]['label']
            score = result[0]['score']
            st.write(f"Sentiment: {sentiment} (Confidence: {score:.2f})")

            sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            if sentiment == "POSITIVE":
                sentiment_counts['Positive'] += 1
            elif sentiment == "NEGATIVE":
                sentiment_counts['Negative'] += 1
            else:
                sentiment_counts['Neutral'] += 1

            plot_sentiment_pie_chart(sentiment_counts)
            '''fig = plot_sentiment_pie_chart(sentiment_counts)
            st.pyplot(fig)'''
        else:
            st.write("Please enter some text for analysis.")


    stock_symbol = st.text_input("Enter Stock Symbol (e.g., TATAMOTORS.NS)", "TATAMOTORS.NS")
    initial_capital = st.number_input("Enter Initial Capital (INR)", min_value=1000, value=100000)

    timeframe_option = st.selectbox("Select Timeframe for Data", ["1 Day", "1 month"])



    if st.button("Run Simulation"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Fetching live data and running simulation...")
        period = '1mo'
        interval = '5m'

        if timeframe_option == "1 Day":
            period = '1d'
            interval = '1m'
        elif timeframe_option == "1 Week":
            period = '1mo'
            interval = '5m'

        data = fetch_live_data(stock_symbol, interval=interval, period=period)
        data = calculate_signals(data)
        st.write("Fetched Data:", data)
        if data.empty:
            st.error("No data fetched. Please check the stock symbol or try a different timeframe.")
        else:
            data = calculate_signals(data)

        portfolio = simulate_portfolio(data, initial_capital)


        st.write(f"Latest data for {stock_symbol}:")
        st.dataframe(data.tail())

        st.write("Portfolio summary:")
        st.dataframe(portfolio.tail())

        st.write(f"Stock Price and Moving Averages ({timeframe_option}):")
        st.pyplot(plot_stock_data(data, portfolio, timeframe_option))

        st.write(f"Portfolio Value Over Time ({timeframe_option}):")
        st.pyplot(plot_profit_chart(portfolio, timeframe_option))

if __name__ == "__main__":
    streamlit_interface()
