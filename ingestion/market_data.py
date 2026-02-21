
import yfinance as yf
import pandas as pd

# Global indices, stocks, forex, crypto — all free
def get_realtime_data(symbols: list):
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data[symbol] = {
            "current": ticker.fast_info,
            "history": ticker.history(period="1d", interval="5m"),
            "info": ticker.info
        }
    return data

# Example symbols
GLOBAL_MARKETS = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones
    "^IXIC",   # NASDAQ
    "^NSEI",   # NIFTY 50 (India)
    "^BSESN",  # SENSEX
    "^FTSE",   # UK
    "^N225",   # Japan Nikkei
    "GC=F",    # Gold
    "CL=F",    # Crude Oil
    "BTC-USD", # Bitcoin
]
