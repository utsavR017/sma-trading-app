# streamlit_sma_trading_skeleton.py
# Beginner-friendly Streamlit app skeleton
# Usage: pip install -r requirements.txt (streamlit yfinance pandas matplotlib)
# Run: streamlit run streamlit_sma_trading_skeleton.py

# streamlit_sma_trading_skeleton.py
# Updated robust Streamlit SMA Trading App

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="SMA Strategy Demo", layout="wide")

st.title("📈 SMA Strategy Explorer — Beginner Friendly")
st.markdown("""A demo app to explore SMA crossover strategies with adjustable parameters.""")

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
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            st.warning("⚠️ No data found for this ticker or date range. Try another ticker.")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# Determine which column to use for prices
def get_price_column(df):
    for col in ['Adj Close', 'Close', 'close']:
        if col in df.columns:
            return col
    return None

# Compute SMA indicators
def compute_indicators(df, short_w, long_w):
    df = df.copy()
    price_col = get_price_column(df)
    if price_col is None:
        st.error("❌ No valid price column found ('Adj Close' or 'Close').")
        return pd.DataFrame()
    df['SMA_short'] = df[price_col].rolling(short_w).mean()
    df['SMA_long'] = df[price_col].rolling(long_w).mean()
    df['price_col_used'] = price_col
    return df

# Generate trading signals
def generate_signals(df):
    df = df.copy()
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df['position'] = df['signal'].shift(1).fillna(0)
    return df

# Backtest strategy
def backtest(df, init_cash, trade_cost):
    df = df.copy()
    price_col = get_price_column(df)
    if price_col is None:
        st.error("❌ No valid price column found for backtesting.")
        return pd.DataFrame()
    df['market_ret'] = df[price_col].pct_change().fillna(0)
    df['strategy_ret'] = df['position'] * df['market_ret']
    df['trade'] = df['position'].diff().abs().fillna(0)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade'] * trade_cost
    df['equity'] = init_cash * (1 + df['strategy_ret_net']).cumprod()
    df['buy_hold_equity'] = init_cash * (1 + df['market_ret']).cumprod()
    return df

# Performance metrics
def cagr(equity):
    total_return = equity.iloc[-1]/equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0: return 0.0
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
            st.error('No data to process.')
        else:
            df = compute_indicators(df, int(short_w), int(long_w))
            if df.empty:
                st.stop()
            df = generate_signals(df)
            df = backtest(df, float(init_cash), float(trade_cost))
            if df.empty:
                st.stop()
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
            price_col = df['price_col_used']
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(df.index, df[price_col], label='Price')
            ax2.plot(df.index, df['SMA_short'], label=f'SMA {short_w}')
            ax2.plot(df.index, df['SMA_long'], label=f'SMA {long_w}')
            ax2.legend()
            st.pyplot(fig2)
            # Recent trades
            st.subheader('Recent trades (signal changes)')
            trades = df[df['trade'] == 1].copy()
            if trades.empty:
                st.write('No trades in the selected period / parameters.')
            else:
                trades_display = trades[[price_col,'SMA_short','SMA_long','position']].copy()
                trades_display['date'] = trades_display.index
                st.dataframe(trades_display.tail(50))
            # Download CSV
            csv = df.to_csv().encode('utf-8')
            st.download_button(label='Download full results (CSV)', data=csv, file_name=f'{ticker}_sma_results.csv', mime='text/csv')
            # Notes
            st.markdown('''
            **Notes:** This is a demo for educational purposes. The strategy does not guarantee real returns.
            ''')
else:
    st.info('Change parameters in the sidebar and click **Run** to download data and run the backtest.')

st.markdown('---')
st.markdown('Made for a college minor project demo.')
