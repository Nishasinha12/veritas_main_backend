import sqlite3

# Connect to your database
conn = sqlite3.connect('veritas.db')
cursor = conn.cursor()

try:
    
    # Drop the second column (replace 'google_email' with your other column name)
    print("Dropping column 'google_name'...")
    cursor.execute("ALTER TABLE users DROP COLUMN name")
    print("Column 'google_name ' dropped successfully.")

    # Commit the changes to the database
    conn.commit()
    print("Database changes have been committed.")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")
    print("Rolling back changes.")
    conn.rollback()

finally:
    # Close the connection
    conn.close()
    print("Database connection closed.")