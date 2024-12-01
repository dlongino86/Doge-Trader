import requests 
import pandas as pd
import ta
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import hmac
import hashlib
import base64
import time

# Coinbase API credentials
COINBASE_API_KEY = "organizations/291efc14-157b-4892-b774-32416255a76b/apiKeys/25489d72-e507-4513-a69b-cad066dec806"
COINBASE_PRIVATE_KEY = """
MHcCAQEEICOv1l57dT4cg+MmJl7mpxS7dWxB8Zvn9gmokX9xfl26oAoGCCqGSM49
AwEHoUQDQgAE9+LNRyRuSfkwOuESD6ciR71oaqmTooRA7TIFfVoysYbeXWo8Kv3j
fr4SVFUqh6AtbPSufQBY2H2JB+UsqQvmcg==
"""

COINBASE_API_URL = "https://api.exchange.coinbase.com"

# Function to generate Coinbase API authentication headers
def get_auth_headers(method, request_path, body=""):
    private_key = COINBASE_PRIVATE_KEY.replace("\n", "").strip()
    decoded_key = base64.b64decode(private_key)

    timestamp = str(int(time.time()))
    message = f"{timestamp}{method}{request_path}{body}"
    signature = hmac.new(decoded_key, message.encode(), hashlib.sha256).digest()
    signature_b64 = base64.b64encode(signature).decode()

    return {
        "CB-ACCESS-KEY": COINBASE_API_KEY,
        "CB-ACCESS-SIGN": signature_b64,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

# Fetch historical data from Coinbase with authentication
def fetch_historical_data(product_id="DOGE-USD", granularity=3600):
    url = f"{COINBASE_API_URL}/products/{product_id}/candles"
    request_path = f"/products/{product_id}/candles"
    headers = get_auth_headers("GET", request_path)
    params = {"granularity": granularity}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")

    data = response.json()
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time")
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.drop(columns=["time"], inplace=True)
    return df

# Add technical indicators
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["SMA20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["SMA50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["Upper_Band"] = bb.bollinger_hband()
    df["Lower_Band"] = bb.bollinger_lband()
    df["MACD"] = ta.trend.MACD(df["close"]).macd()
    df["MACD_Signal"] = ta.trend.MACD(df["close"]).macd_signal()
    return df

# Simulate trades
def simulate_trades(df):
    initial_balance = 1000
    balance = initial_balance
    position = 0

    for i in range(1, len(df)):
        if df["Signal"].iloc[i] == 1:  # Buy signal
            if balance > 0:
                position = balance / df["close"].iloc[i]
                balance = 0
        elif df["Signal"].iloc[i] == -1:  # Sell signal
            if position > 0:
                balance = position * df["close"].iloc[i]
                position = 0

    balance += position * df["close"].iloc[-1]
    return balance - initial_balance

# Apply trading strategy
def apply_strategy(df, strategy="rsi"):
    df["Signal"] = 0
    if strategy == "rsi":
        df.loc[df["RSI"] < 30, "Signal"] = 1
        df.loc[df["RSI"] > 70, "Signal"] = -1
    elif strategy == "moving_average":
        df.loc[df["SMA20"] > df["SMA50"], "Signal"] = 1
        df.loc[df["SMA20"] < df["SMA50"], "Signal"] = -1
    elif strategy == "bollinger_bands":
        df.loc[df["close"] < df["Lower_Band"], "Signal"] = 1
        df.loc[df["close"] > df["Upper_Band"], "Signal"] = -1
    elif strategy == "macd":
        df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1
        df.loc[df["MACD"] < df["MACD_Signal"], "Signal"] = -1
    return df

# Backtest strategies
def backtest_strategy(df, strategy_function, params):
    results = {}
    for param in params:
        strategy_df = strategy_function(df.copy(), param)
        profit = simulate_trades(strategy_df)
        results[param] = profit
    return results

# Train LSTM model
def train_lstm_model(df):
    data = df["close"].values.reshape(-1, 1)
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    lookback = 10
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    return model, lookback, np.min(data), np.max(data)

# Predict prices
def predict_prices(model, df, lookback, data_min, data_max):
    data = df["close"].values.reshape(-1, 1)
    scaled_data = (data - data_min) / (data_max - data_min)
    X_test = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction = model.predict(X_test)
    return prediction[0][0] * (data_max - data_min) + data_min

# Streamlit dashboard
def streamlit_dashboard():
    st.title("Dogecoin Trading Bot with AI and Backtesting")
    st.subheader("Real-Time Analysis, AI Predictions, and Backtesting")

    try:
        df = fetch_historical_data()
        df = add_indicators(df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    model, lookback, data_min, data_max = train_lstm_model(df)
    predicted_price = predict_prices(model, df, lookback, data_min, data_max)
    st.write(f"AI Predicted Next Price: ${predicted_price:.4f}")

    strategies = ["rsi", "moving_average", "bollinger_bands", "macd"]
    params = range(10, 50, 5)
    for strategy in strategies:
        results = backtest_strategy(df, apply_strategy, params)
        st.write(f"Backtest Results for {strategy.capitalize()} Strategy:")
        st.bar_chart(pd.Series(results))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))
    fig.update_layout(title="Dogecoin Live Price and SMA", xaxis_title="Timestamp", yaxis_title="Price (USD)")
    st.plotly_chart(fig)

if __name__ == "__main__":
    streamlit_dashboard()
