"""
Database migration script to add new columns to the applications table.
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Database connection parameters
DB_USER = os.getenv("DB_USER", "amarjeet")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "social")

# Create database URL
if DB_PASSWORD:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def migrate_database():
    """Add new columns to the applications table."""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Connect to the database
        with engine.connect() as connection:
            # Check if columns exist before adding them
            columns_to_add = [
                {"name": "name", "type": "VARCHAR(255)", "nullable": "NULL"},
                {"name": "email", "type": "VARCHAR(255)", "nullable": "NULL"},
                {"name": "phone", "type": "VARCHAR(50)", "nullable": "NULL"},
                {"name": "emirates_id", "type": "VARCHAR(50)", "nullable": "NULL"},
                {"name": "employment_status", "type": "VARCHAR(50)", "nullable": "NULL"},
                {"name": "employer", "type": "VARCHAR(255)", "nullable": "NULL"},
                {"name": "job_title", "type": "VARCHAR(255)", "nullable": "NULL"},
                {"name": "employment_duration", "type": "INTEGER", "nullable": "NULL"},
                {"name": "monthly_expenses", "type": "FLOAT", "nullable": "NULL"},
                {"name": "assets_value", "type": "FLOAT", "nullable": "NULL"},
                {"name": "liabilities_value", "type": "FLOAT", "nullable": "NULL"}
            ]
            
            for column in columns_to_add:
                # Check if column exists
                check_query = text(f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='applications' AND column_name='{column['name']}'
                );
                """)
                
                result = connection.execute(check_query).fetchone()
                
                # If column doesn't exist, add it
                if not result[0]:
                    print(f"Adding column '{column['name']}' to applications table...")
                    alter_query = text(f"""
                    ALTER TABLE applications
                    ADD COLUMN {column['name']} {column['type']} {column['nullable']};
                    """)
                    connection.execute(alter_query)
                    connection.commit()
                else:
                    print(f"Column '{column['name']}' already exists in applications table.")
            
            print("Database migration completed successfully.")
            return True
    except Exception as e:
        print(f"Error migrating database: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting database migration...")
    success = migrate_database()
    sys.exit(0 if success else 1)
