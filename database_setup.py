# backend/database_setup.py
import sqlite3
conn = sqlite3.connect('veritas.db') 
cursor = conn.cursor()
print("Database connected. Creating table...")
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id BIGINT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
print("Table 'users' created successfully.")
conn.close()