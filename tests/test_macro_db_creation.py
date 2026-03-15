import sqlite3
import os

DB_PATH = "/home/kali/Downloads/Qual/data/sql_db/macro.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS macro_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_name TEXT NOT NULL,
        series_id TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        value REAL NOT NULL,
        UNIQUE(series_id, observation_date)
    )
    """)
    
    conn.commit()
    conn.close()
    print("Macro DB created!")

if __name__ == "__main__":
    create_db()
