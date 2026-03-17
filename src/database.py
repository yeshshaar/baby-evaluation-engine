import sqlite3
import pandas as pd
import os

DB_PATH = "data/yield_engine.db"

def init_db():
    """Initializes the database and forces a schema check."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations';")
    table_exists = cursor.fetchone()

    if table_exists:
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = [column[1] for column in cursor.fetchall()]

        # Drop if any expected column is missing
        expected = {"Candidate Name", "Score", "Model Used"}
        if not expected.issubset(set(columns)):
            print("⚠️ Stale database schema detected. Dropping old table...")
            cursor.execute("DROP TABLE evaluations")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            "Candidate Name"        TEXT,
            "Score"                 REAL,
            "Skill Match"           REAL,
            "Semantic Match"        REAL,
            "Experience Relevance"  REAL,
            "Matched Skills"        TEXT,
            "Missing Skills"        TEXT,
            "Model Used"            TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_evaluation(data_list):
    """Saves multiple evaluation records into the database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(data_list)
    df.to_sql("evaluations", conn, if_exists="append", index=False)
    conn.close()

def get_all_evaluations():
    """Retrieves all history from the database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY timestamp DESC", conn)
    conn.close()
    return df
