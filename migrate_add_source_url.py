"""
Add source_url field to jobs table for storing original URLs
"""
from sqlalchemy import text
from database import engine

def migrate():
    """Add source_url column to jobs table"""
    with engine.connect() as conn:
        # Add the new column
        conn.execute(text("ALTER TABLE jobs ADD COLUMN source_url TEXT"))
        conn.commit()
        print("✅ Added source_url column to jobs table")

if __name__ == "__main__":
    migrate()
