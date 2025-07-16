#!/usr/bin/env python3
"""
Database migration script to add user preferences table
"""

import sqlite3
import os
from datetime import datetime

DATABASE_PATH = "video_audio_service.db"

def migrate_add_media_hash_column():
    """Add media_hash column to jobs table if it doesn't exist"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if media_hash column already exists
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'media_hash' not in columns:
            print("🔧 Adding media_hash column to jobs table...")
            cursor.execute("ALTER TABLE jobs ADD COLUMN media_hash TEXT")
            cursor.execute("CREATE INDEX idx_jobs_media_hash ON jobs(media_hash)")
            conn.commit()
            print("✅ Successfully added media_hash column")
        else:
            print("✅ media_hash column already exists")
    except Exception as e:
        print(f"❌ Error adding media_hash column: {e}")
        conn.rollback()
    finally:
        conn.close()

def migrate_add_preferences_table():
    """Add user_preferences table to the database"""
    if not os.path.exists(DATABASE_PATH):
        print(f"Database {DATABASE_PATH} not found. Creating new database with preferences table...")
        # The table will be created automatically when we import database.py
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user_preferences table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
        if cursor.fetchone():
            print("✅ user_preferences table already exists")
            return
        
        print("🔧 Creating user_preferences table...")
        
        # Create user_preferences table
        cursor.execute("""
            CREATE TABLE user_preferences (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR UNIQUE,
                preferred_voice VARCHAR DEFAULT 'en-US-AriaNeural',
                voice_speed VARCHAR DEFAULT '1.0',
                voice_pitch VARCHAR DEFAULT '0Hz',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create index on user_id
        cursor.execute("CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id)")
        
        conn.commit()
        print("✅ Successfully created user_preferences table")
        
        # Create default preferences for existing users
        cursor.execute("SELECT id FROM users")
        existing_users = cursor.fetchall()
        
        if existing_users:
            print(f"🔧 Creating default preferences for {len(existing_users)} existing users...")
            import uuid
            for (user_id,) in existing_users:
                pref_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO user_preferences (id, user_id, preferred_voice, voice_speed, voice_pitch)
                    VALUES (?, ?, 'en-US-AriaNeural', '1.0', '0Hz')
                """, (pref_id, user_id))
            conn.commit()
            print("✅ Created default preferences for existing users")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Starting database migration...")
    migrate_add_media_hash_column()
    migrate_add_preferences_table()
    print("✅ Migration completed!")
