# streamlit_sma_trading_skeleton.py
# Beginner-friendly Streamlit app skeleton
# Usage: pip install -r requirements.txt (streamlit yfinance pandas matplotlib)
# Run: streamlit run streamlit_sma_trading_skeleton.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="SMA Strategy Demo", layout="wide")

st.title("📈 SMA Strategy Explorer — Beginner Friendly")
st.markdown(
    """A simple demo app that downloads historical price data, builds SMA crossover signals,
    backtests a basic strategy, and shows equity curves. Use the sidebar to change parameters
    (SMA windows, trade costs, ticker, etc.) and see results update instantly.
    """
)

# ---------------------- Sidebar controls ----------------------
with st.sidebar.form(key='controls'):
    ticker = st.text_input('Ticker', value='AAPL')
    start_date = st.date_input('Start date', value=pd.to_datetime('2018-01-01'))
    end_date = st.date_input('End date', value=datetime.today())

    short_w = st.number_input('Short SMA window', min_value=2, max_value=200, value=10)
    long_w = st.number_input('Long SMA window', min_value=3, max_value=400, value=50)
    init_cash = st.number_input('Initial cash (USD)', min_value=100.0, value=10000.0, step=100.0)
    trade_cost = st.number_input('Per-trade cost (fraction, e.g. 0.001 = 0.1%)', min_value=0.0, value=0.001, step=0.0005, format="%.6f")

    run_button = st.form_submit_button('Run')

# ---------------------- Helper functions ----------------------
@st.cache_data(ttl=60*60)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    df = df[['Open','High','Low','Close','Adj Close','Volume']]
    df = df.dropna()
    return df


def compute_indicators(df, short_w, long_w):
    df = df.copy()
    df['SMA_short'] = df['Adj Close'].rolling(short_w).mean()
    df['SMA_long']  = df['Adj Close'].rolling(long_w).mean()
    return df


def generate_signals(df):
    df = df.copy()
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df['position'] = df['signal'].shift(1).fillna(0)  # act next day
    return df


def backtest(df, init_cash, trade_cost):
    df = df.copy()
    df['market_ret'] = df['Adj Close'].pct_change().fillna(0)
    df['strategy_ret'] = df['position'] * df['market_ret']
    df['trade'] = df['position'].diff().abs().fillna(0)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade'] * trade_cost
    df['equity'] = init_cash * (1 + df['strategy_ret_net']).cumprod()
    df['buy_hold_equity'] = init_cash * (1 + df['market_ret']).cumprod()
    return df


def cagr(equity):
    total_return = equity.iloc[-1]/equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1/years) - 1


def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()

# ---------------------- Main run ----------------------
if run_button:
    with st.spinner('Downloading data and running backtest...'):
        df = load_data(ticker, start_date, end_date)
        if df.empty:
            st.error('No data found for ticker or date range. Try another ticker or earlier start date.')
        else:
            df = compute_indicators(df, int(short_w), int(long_w))
            df = generate_signals(df)
            df = backtest(df, float(init_cash), float(trade_cost))

            # Metrics
            strat_cagr = cagr(df['equity'])
            bh_cagr = cagr(df['buy_hold_equity'])
            strat_mdd = max_drawdown(df['equity'])

            col1, col2, col3 = st.columns(3)
            col1.metric('Strategy CAGR', f"{strat_cagr:.2%}")
            col2.metric('Buy & Hold CAGR', f"{bh_cagr:.2%}")
            col3.metric('Strategy Max Drawdown', f"{strat_mdd:.2%}")

            # Equity curve plot
            st.subheader('Equity Curve')
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(df.index, df['equity'], label='SMA Strategy')
            ax.plot(df.index, df['buy_hold_equity'], label='Buy & Hold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value (USD)')
            ax.legend()
            st.pyplot(fig)

            # Price + SMAs
            st.subheader('Price and SMAs')
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(df.index, df['Adj Close'], label='Adj Close')
            ax2.plot(df.index, df['SMA_short'], label=f'SMA {short_w}')
            ax2.plot(df.index, df['SMA_long'], label=f'SMA {long_w}')
            ax2.legend()
            st.pyplot(fig2)

            # Show recent trades
            st.subheader('Recent trades (signal changes)')
            trades = df[df['trade'] == 1].copy()
            if trades.empty:
                st.write('No trades in the selected period / parameters.')
            else:
                trades_display = trades[['Adj Close','SMA_short','SMA_long','position']].copy()
                trades_display['date'] = trades_display.index
                st.dataframe(trades_display.tail(50))

            # Download CSV
            csv = df.to_csv().encode('utf-8')
            st.download_button(label='Download full results (CSV)', data=csv, file_name=f'{ticker}_sma_results.csv', mime='text/csv')

            # Short explanation for beginners
            st.markdown('''
            **Notes for beginners**
            - This is a simple academic demo: it **does not** represent financial advice.
            - The strategy is: go long when the short SMA is above the long SMA; otherwise stay in cash.
            - Trade costs are subtracted on days when the position changes (buy or sell).
            - Use longer data ranges for more reliable statistics and always validate on out-of-sample data.
            ''')

else:
    st.info('Change parameters in the sidebar and click **Run** to download data and run the backtest.')

# Footer / resources
st.markdown('---')
st.markdown('Made for a college minor project demo. For upgrades: add ML, risk controls, paper trading via Alpaca, and a nicer dashboard.')
