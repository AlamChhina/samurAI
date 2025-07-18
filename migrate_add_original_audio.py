#!/usr/bin/env python3
"""
Migration script to add original_audio_file column to the jobs table
"""
import sqlite3
import os

def migrate_database():
    db_path = "video_audio_service.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist. Nothing to migrate.")
        return
    
    print(f"Migrating database: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'original_audio_file' in columns:
            print("Column 'original_audio_file' already exists. No migration needed.")
            return
        
        # Add the new column
        cursor.execute("ALTER TABLE jobs ADD COLUMN original_audio_file TEXT")
        
        # Commit the changes
        conn.commit()
        print("✅ Successfully added 'original_audio_file' column to jobs table")
        
        # Show updated schema
        cursor.execute("PRAGMA table_info(jobs)")
        columns = cursor.fetchall()
        print("\nUpdated jobs table schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
    except sqlite3.Error as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
