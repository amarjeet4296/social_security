"""
Database migration script to add new columns to the documents table.
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

def migrate_documents_table():
    """Add new columns to the documents table and create column aliases."""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Connect to the database
        with engine.connect() as connection:
            # Add new columns if they don't exist
            columns_to_add = [
                {"name": "document_type", "type": "VARCHAR(50)", "nullable": "NULL"},
                {"name": "file_path", "type": "TEXT", "nullable": "NULL"},
                {"name": "mime_type", "type": "VARCHAR(100)", "nullable": "NULL"},
                {"name": "uploaded_at", "type": "TIMESTAMP", "nullable": "NULL DEFAULT NOW()"}
            ]
            
            for column in columns_to_add:
                # Check if column exists
                check_query = text(f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='documents' AND column_name='{column['name']}'
                );
                """)
                
                result = connection.execute(check_query).fetchone()
                
                # If column doesn't exist, add it
                if not result[0]:
                    print(f"Adding column '{column['name']}' to documents table...")
                    alter_query = text(f"""
                    ALTER TABLE documents
                    ADD COLUMN {column['name']} {column['type']} {column['nullable']};
                    """)
                    connection.execute(alter_query)
                    connection.commit()
                else:
                    print(f"Column '{column['name']}' already exists in documents table.")
            
            # Update existing records to match new schema
            # Set document_type based on file_type for existing records
            check_column_query = text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name='documents' AND column_name='file_type'
            );
            """)
            
            has_file_type = connection.execute(check_column_query).fetchone()[0]
            
            if has_file_type:
                print("Updating document_type values based on file_type...")
                update_query = text("""
                UPDATE documents 
                SET document_type = file_type 
                WHERE document_type IS NULL AND file_type IS NOT NULL;
                """)
                connection.execute(update_query)
                connection.commit()
            
            print("Documents table migration completed successfully.")
            return True
            
    except Exception as e:
        print(f"Error migrating documents table: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting documents table migration...")
    success = migrate_documents_table()
    sys.exit(0 if success else 1)
