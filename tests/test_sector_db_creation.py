import sqlite3
import os

DB_PATH = "/home/kali/Downloads/Qual/data/sql_db/sectors.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_db():
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
    conn.close()
    print("Sectors DB created!")

if __name__ == "__main__":
    create_db()
