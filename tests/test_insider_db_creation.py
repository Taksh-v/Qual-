import sqlite3
import os

DB_PATH = "/home/kali/Downloads/Qual/data/sql_db/insider.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS insider_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        filing_date TEXT NOT NULL,
        reporting_owner TEXT NOT NULL,
        transaction_type TEXT NOT NULL,
        shares_traded REAL NOT NULL,
        transaction_price REAL NOT NULL,
        value_traded REAL NOT NULL,
        accession_number TEXT NOT NULL UNIQUE
    )
    """)
    
    conn.commit()
    conn.close()
    print("Insider DB created!")

if __name__ == "__main__":
    create_db()
