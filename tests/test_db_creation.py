import sqlite3
import os

DB_DIR = "data/sql_db"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "fundamentals.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the Income Statement table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS income_statement (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        fiscal_quarter INTEGER,
        report_date TEXT NOT NULL,
        revenue REAL,
        gross_profit REAL,
        operating_income REAL,
        net_income REAL,
        eps REAL,
        UNIQUE(ticker, fiscal_year, fiscal_quarter)
    )
    ''')
    
    # Create Balance Sheet table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS balance_sheet (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        fiscal_quarter INTEGER,
        report_date TEXT NOT NULL,
        total_assets REAL,
        total_liabilities REAL,
        total_equity REAL,
        cash_and_equivalents REAL,
        total_debt REAL,
        UNIQUE(ticker, fiscal_year, fiscal_quarter)
    )
    ''')
    
    # Create Cash Flow table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cash_flow (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        fiscal_quarter INTEGER,
        report_date TEXT NOT NULL,
        operating_cash_flow REAL,
        investing_cash_flow REAL,
        financing_cash_flow REAL,
        free_cash_flow REAL,
        UNIQUE(ticker, fiscal_year, fiscal_quarter)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Created SQLite Fundamentals schema at {DB_PATH}")

if __name__ == "__main__":
    init_db()
