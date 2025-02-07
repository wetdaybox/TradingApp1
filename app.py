#!/usr/bin/env python3
"""
Autonomous Adaptive Trading System – Streamlit Version

This web app uses free historical data for AAPL (from Yahoo Finance via yfinance),
calculates a 50-day SMA signal, simulates a leveraged trading strategy with dynamic position sizing,
and displays interactive plots along with a detailed trade recommendation.

DISCLAIMER:
  This system is experimental and uses aggressive leverage and risk management.
  No trading system is foolproof. This code is provided solely for educational and experimental purposes.
  Do not use this system with real funds without thorough testing and risk assessment.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from datetime import datetime, timedelta

# --- Core Trading Functions ---

def fetch_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch historical daily data for the given stock symbol from Yahoo Finance.
    Returns a DataFrame with the 'Adj Close' prices.
    """
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("No data fetched. Check the stock symbol and date range.")
        return pd.DataFrame()
    df = data[['Adj Close']].rename(columns={'Adj Close': 'price'})
    return df

def calculate_sma(series, window):
    """Calculate the Simple Moving Average (SMA) for a pandas Series."""
    return series.rolling(window=window, min_periods=1).mean()

def generate_signal(df, sma_window=50):
    """
    Generate a binary signal: 1 if price > SMA, else 0.
    The signal is shifted by one day to avoid lookahead bias.
    """
    df['SMA'] = calculate_sma(df['price'], sma_window)
    df['signal'] = np.where(df['price'] > df['SMA'], 1, 0)
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def dynamic_position_size(current_price, stop_loss_pct, portfolio_value, risk_pct=0.01):
    """
    Calculate the number of shares so that the risk per trade (difference between price and stop-loss)
    is only risk_pct of the portfolio.
    """
    risk_per_share = current_price * stop_loss_pct
    if risk_per_share <= 0:
        return 0
    risk_amount = portfolio_value * risk_pct
    shares = floor(risk_amount / risk_per_share)
    return shares

def calculate_trade_recommendation(df, portfolio_value=10000, leverage=5, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Based on the latest available data, if signal == 1 then recommend a BUY trade.
    Uses dynamic position sizing and applies a leverage multiplier.
    """
    latest = df.iloc[-1]
    current_price = latest['price']
    signal = latest['signal']
    if signal == 1:
        base_shares = dynamic_position_size(current_price, stop_loss_pct, portfolio_value)
        leveraged_shares = base_shares * leverage
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        recommendation = {
            'action': 'BUY',
            'stock': 'AAPL',
            'current_price': current_price,
            'num_shares': leveraged_shares,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    else:
        recommendation = {
            'action': 'HOLD/NO POSITION',
            'stock': 'AAPL',
            'current_price': current_price
        }
    return recommendation

def simulate_leveraged_cumulative_return(df, leverage=5):
    """
    Simulate cumulative return for the leveraged strategy.
    When the signal is 1, daily returns are amplified by the leverage factor.
    """
    df['daily_return'] = df['price'].pct_change().fillna(0)
    df['strategy_return'] = leverage * df['daily_return'] * df['signal']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def plot_results(df, stock_symbol, start_date, end_date):
    """
    Create a two-panel plot:
     - Upper panel: Stock price and 50-day SMA.
     - Lower panel: Cumulative leveraged return.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    ax1.plot(df.index, df['price'], label='Price', color='black')
    ax1.plot(df.index, df['SMA'], label='50-day SMA', color='blue', linestyle='--')
    ax1.set_title(f"{stock_symbol} Price and 50-day SMA\n({start_date} to {end_date})")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df.index, df['cumulative_return'], label='Cumulative Leveraged Return', color='green')
    ax2.set_title("Cumulative Strategy Return")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig)

# --- Persistent Data Storage ---
def save_results(df, filename="trading_results.csv"):
    """
    Save the latest simulation results (timestamp, current price, cumulative return)
    to a CSV file for persistence.
    """
    result = pd.DataFrame({
        "timestamp": [datetime.now()],
        "current_price": [df['price'].iloc[-1]],
        "cumulative_return": [df['cumulative_return'].iloc[-1]]
    })
    if os.path.isfile(filename):
        result.to_csv(filename, mode="a", header=False, index=False)
    else:
        result.to_csv(filename, index=False)

# --- Main Trading Cycle ---
def run_trading_cycle():
    stock_symbol = 'AAPL'
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    portfolio_value = 10000
    leverage = 5
    sma_window = 50

    df = fetch_stock_data(stock_symbol, start_date, end_date)
    df = generate_signal(df, sma_window=sma_window)
    df = simulate_leveraged_cumulative_return(df, leverage=leverage)
    recommendation = calculate_trade_recommendation(df, portfolio_value, leverage)
    save_results(df)
    return df, recommendation, start_date, end_date

# --- Streamlit App Interface ---
st.set_page_config(page_title="Autonomous Adaptive Trading System", layout="wide")
st.title("Autonomous Adaptive Trading System")
st.markdown("""
This system fetches free, up‑to‑date AAPL data, calculates a 50‑day SMA trend signal,
simulates an aggressive leveraged strategy with adaptive position sizing, and displays
an interactive plot and trade recommendation.

**DISCLAIMER:** This system is experimental and uses leverage. No system is foolproof.
Use with extreme caution and only for educational purposes.
""")

if st.button("Run Trading Simulation"):
    with st.spinner("Fetching data and running simulation..."):
        try:
            df, rec, start_date, end_date = run_trading_cycle()
            plot_results(df, "AAPL", start_date, end_date)
            if rec['action'] == 'BUY':
                st.success(f"Trade Recommendation for AAPL:\n"
                           f"Action: BUY\n"
                           f"Current Price: ${rec['current_price']:.2f}\n"
                           f"Buy {rec['num_shares']} shares using {rec['leverage']}x leverage\n"
                           f"Stop-Loss: ${rec['stop_loss']:.2f}\n"
                           f"Take-Profit: ${rec['take_profit']:.2f}")
            else:
                st.info(f"Trade Recommendation for AAPL:\n"
                        f"Action: HOLD/NO POSITION\n"
                        f"Current Price: ${rec['current_price']:.2f}")
        except Exception as e:
            st.error(f"Error during simulation: {e}")

st.markdown("---")
st.markdown("### Persistent Data")
if st.button("Show Saved Results"):
    if os.path.isfile("trading_results.csv"):
        saved_df = pd.read_csv("trading_results.csv")
        st.dataframe(saved_df)
    else:
        st.info("No saved results found.")

st.markdown("### About")
st.markdown("""
This application is designed to run autonomously and is updated using free historical data.
Each run fetches the most up‑to‑date data (using your system's current date) and then executes a full
simulation of the trading strategy. All results are saved to a CSV file for later review, and the system
displays both visual plots and a clear trade recommendation.
""")

# --- Unit Tests Section (Hidden) ---
if "run_tests" in st.experimental_get_query_params():
    st.write("Running unit tests...")
    class TestTradingFunctions(unittest.TestCase):
        def setUp(self):
            dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
            prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], index=dates)
            self.df_test = pd.DataFrame({'price': prices})
    
        def test_calculate_sma(self):
            sma = calculate_sma(self.df_test['price'], window=3)
            expected = (100 + 102 + 101) / 3
            self.assertAlmostEqual(sma.iloc[2], expected)
    
        def test_generate_signal(self):
            df_signal = generate_signal(self.df_test.copy(), sma_window=3)
            self.assertIn('signal', df_signal.columns)
    
        def test_trade_recommendation(self):
            df_signal = generate_signal(self.df_test.copy(), sma_window=3)
            rec = calculate_trade_recommendation(df_signal, portfolio_value=10000, leverage=5)
            self.assertIn('action', rec)
            self.assertIn('stock', rec)
            self.assertIn('current_price', rec)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingFunctions)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    st.text("Unit Test Results:")
    st.text(result)
