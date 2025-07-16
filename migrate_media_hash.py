#!/usr/bin/env python3
"""
Migration script to add media_hash column to jobs table
"""

import sqlite3
import os

def migrate_database():
    db_path = "video_audio_service.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if media_hash column already exists
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'media_hash' in columns:
            print("✅ media_hash column already exists in jobs table")
            conn.close()
            return True
        
        # Add media_hash column
        print("📝 Adding media_hash column to jobs table...")
        cursor.execute("ALTER TABLE jobs ADD COLUMN media_hash TEXT")
        
        # Create index on media_hash
        print("📝 Creating index on media_hash column...")
        cursor.execute("CREATE INDEX idx_jobs_media_hash ON jobs(media_hash)")
        
        # Commit changes
        conn.commit()
        print("✅ Successfully added media_hash column and index to jobs table")
        
        # Show updated table structure
        cursor.execute("PRAGMA table_info(jobs)")
        columns = cursor.fetchall()
        print("\n📋 Updated jobs table structure:")
        for column in columns:
            print(f"  - {column[1]} ({column[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    print("🔄 Starting database migration for media_hash column...")
    success = migrate_database()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("   The service should now work properly.")
    else:
        print("\n💥 Migration failed!")
        print("   Please check the error messages above.")
