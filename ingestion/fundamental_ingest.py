import sqlite3
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_DIR = "data/sql_db"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "fundamentals.db")

# Default symbols if none provided
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "GS", "MS", "WFC",
    "JNJ", "UNH", "PFE", "ABBV",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP"
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the Income Statement table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS income_statement (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        report_date TEXT NOT NULL,
        revenue REAL,
        gross_profit REAL,
        operating_income REAL,
        net_income REAL,
        eps REAL,
        UNIQUE(ticker, fiscal_year)
    )
    ''')
    
    # Create Balance Sheet table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS balance_sheet (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        report_date TEXT NOT NULL,
        total_assets REAL,
        total_liabilities REAL,
        total_equity REAL,
        cash_and_equivalents REAL,
        total_debt REAL,
        UNIQUE(ticker, fiscal_year)
    )
    ''')
    
    # Create Cash Flow table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cash_flow (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        report_date TEXT NOT NULL,
        operating_cash_flow REAL,
        investing_cash_flow REAL,
        financing_cash_flow REAL,
        free_cash_flow REAL,
        UNIQUE(ticker, fiscal_year)
    )
    ''')
    
    conn.commit()
    conn.close()

def _safe_float(val):
    if pd.isna(val):
        return None
    try:
        return float(val)
    except:
        return None

def fetch_and_store_fundamentals(symbols=DEFAULT_SYMBOLS):
    """
    Fetches annual financial statements from Yahoo Finance and stores them in SQLite.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    success_count = 0
    
    for symbol in symbols:
        logger.info(f"Fetching fundamentals for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            
            # --- Income Statement ---
            financials = ticker.financials
            if not financials.empty:
                for col in financials.columns:
                    report_date = str(col)[:10]
                    year = col.year
                    
                    try:
                        revenue = _safe_float(financials.loc['Total Revenue', col]) if 'Total Revenue' in financials.index else None
                        gross_profit = _safe_float(financials.loc['Gross Profit', col]) if 'Gross Profit' in financials.index else None
                        operating_income = _safe_float(financials.loc['Operating Income', col]) if 'Operating Income' in financials.index else None
                        net_income = _safe_float(financials.loc['Net Income', col]) if 'Net Income' in financials.index else None
                        eps = _safe_float(financials.loc['Basic EPS', col]) if 'Basic EPS' in financials.index else None
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO income_statement 
                        (ticker, fiscal_year, report_date, revenue, gross_profit, operating_income, net_income, eps)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, year, report_date, revenue, gross_profit, operating_income, net_income, eps))
                    except Exception as e:
                        logger.warning(f"  Missing Income Statement data for {symbol} ({year}): {e}")

            # --- Balance Sheet ---
            balance_sheet = ticker.balance_sheet
            if not balance_sheet.empty:
                for col in balance_sheet.columns:
                    report_date = str(col)[:10]
                    year = col.year
                    
                    try:
                        total_assets = _safe_float(balance_sheet.loc['Total Assets', col]) if 'Total Assets' in balance_sheet.index else None
                        total_liabilities = _safe_float(balance_sheet.loc['Total Liabilities Net Minority Interest', col]) if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                        total_equity = _safe_float(balance_sheet.loc['Stockholders Equity', col]) if 'Stockholders Equity' in balance_sheet.index else None
                        cash = _safe_float(balance_sheet.loc['Cash And Cash Equivalents', col]) if 'Cash And Cash Equivalents' in balance_sheet.index else None
                        debt = _safe_float(balance_sheet.loc['Total Debt', col]) if 'Total Debt' in balance_sheet.index else None
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO balance_sheet 
                        (ticker, fiscal_year, report_date, total_assets, total_liabilities, total_equity, cash_and_equivalents, total_debt)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, year, report_date, total_assets, total_liabilities, total_equity, cash, debt))
                    except Exception as e:
                        logger.warning(f"  Missing Balance Sheet data for {symbol} ({year}): {e}")
                        
            # --- Cash Flow ---
            cashflow = ticker.cashflow
            if not cashflow.empty:
                for col in cashflow.columns:
                    report_date = str(col)[:10]
                    year = col.year
                    
                    try:
                        op_cf = _safe_float(cashflow.loc['Operating Cash Flow', col]) if 'Operating Cash Flow' in cashflow.index else None
                        inv_cf = _safe_float(cashflow.loc['Investing Cash Flow', col]) if 'Investing Cash Flow' in cashflow.index else None
                        fin_cf = _safe_float(cashflow.loc['Financing Cash Flow', col]) if 'Financing Cash Flow' in cashflow.index else None
                        fcf = _safe_float(cashflow.loc['Free Cash Flow', col]) if 'Free Cash Flow' in cashflow.index else None
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO cash_flow 
                        (ticker, fiscal_year, report_date, operating_cash_flow, investing_cash_flow, financing_cash_flow, free_cash_flow)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (symbol, year, report_date, op_cf, inv_cf, fin_cf, fcf))
                    except Exception as e:
                        logger.warning(f"  Missing Cash Flow data for {symbol} ({year}): {e}")

            success_count += 1
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    conn.close()
    logger.info(f"Successfully updated fundamentals for {success_count} companies.")

if __name__ == "__main__":
    fetch_and_store_fundamentals()
