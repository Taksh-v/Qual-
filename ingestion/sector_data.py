import os
import sqlite3
import yfinance as yf
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "sql_db", "sectors.db")

# Top sector ETFs representing market rotation
SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

def fetch_and_store_sector_data():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sector_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sector_name TEXT NOT NULL,
        etf_ticker TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        price REAL NOT NULL,
        daily_change_pct REAL,
        volume INTEGER,
        UNIQUE(etf_ticker, observation_date)
    )
    """)
    conn.commit()

    success_count = 0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for sector_name, ticker in SECTORS.items():
        try:
            ticker_obj = yf.Ticker(ticker)
            # Fetch 2 days of history to calculate daily change precisely if possible
            hist = ticker_obj.history(period="5d")
            if hist.empty:
                logger.warning(f"No recent data for {ticker}")
                continue
            
            # Use real date of the last close
            last_date_str = hist.index[-1].strftime("%Y-%m-%d")
            
            latest_close = float(hist['Close'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
            
            change_pct = None
            if len(hist) > 1:
                prev_close = float(hist['Close'].iloc[-2])
                if prev_close > 0:
                    change_pct = ((latest_close - prev_close) / prev_close) * 100.0

            cursor.execute("""
                INSERT OR IGNORE INTO sector_performance 
                (sector_name, etf_ticker, observation_date, price, daily_change_pct, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sector_name, ticker, last_date_str, latest_close, change_pct, volume))
            
            if cursor.rowcount > 0:
                success_count += 1
                logger.debug(f"Fetched {sector_name} ({ticker}): ${latest_close:.2f} ({change_pct:.2f}%) on {last_date_str}")
            
        except Exception as e:
            logger.error(f"Failed fetching {sector_name} ({ticker}): {e}")
            
    conn.commit()
    conn.close()
    
    logger.info(f"Stored {success_count} new sector readings in SQLite DB.")
    return True

if __name__ == "__main__":
    fetch_and_store_sector_data()
