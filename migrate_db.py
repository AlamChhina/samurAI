#!/usr/bin/env python3
"""
Database migration script to add summary-related columns to existing jobs table.
This script is idempotent - it can be run multiple times safely.
"""

import sqlite3
import os
from pathlib import Path

DATABASE_PATH = "video_audio_service.db"

def migrate_database():
    """Add summary-related columns to jobs table if they don't exist"""
    if not os.path.exists(DATABASE_PATH):
        print(f"Database {DATABASE_PATH} not found. No migration needed.")
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get current schema
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Check which columns need to be added
        columns_to_add = []
        if 'summary' not in columns:
            columns_to_add.append(('summary', 'TEXT'))
        if 'summary_file' not in columns:
            columns_to_add.append(('summary_file', 'VARCHAR'))
        if 'summary_speech_file' not in columns:
            columns_to_add.append(('summary_speech_file', 'VARCHAR'))
        if 'summary_prompt' not in columns:
            columns_to_add.append(('summary_prompt', 'TEXT'))
        
        # Add missing columns
        for column_name, column_type in columns_to_add:
            print(f"Adding column: {column_name} {column_type}")
            cursor.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}")
        
        if columns_to_add:
            conn.commit()
            print(f"Successfully added {len(columns_to_add)} columns to jobs table")
        else:
            print("No migration needed - all columns already exist")
            
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
