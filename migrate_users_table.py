#!/usr/bin/env python3
"""
Database migration script to add subscription-related columns to users table
"""

import sqlite3
from datetime import datetime

def migrate_users_table():
    """Add subscription-related columns to users table"""
    
    # Connect to database
    conn = sqlite3.connect('video_audio_service.db')
    cursor = conn.cursor()
    
    try:
        print("🔄 Starting users table migration...")
        
        # Add subscription_tier column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_tier VARCHAR DEFAULT "free"')
            print("✅ Added subscription_tier column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ subscription_tier column already exists")
            else:
                raise
        
        # Add stripe_customer_id column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR')
            print("✅ Added stripe_customer_id column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ stripe_customer_id column already exists")
            else:
                raise
        
        # Add subscription_status column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_status VARCHAR DEFAULT "inactive"')
            print("✅ Added subscription_status column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ subscription_status column already exists")
            else:
                raise
        
        # Add subscription_start_date column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_start_date DATETIME')
            print("✅ Added subscription_start_date column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ subscription_start_date column already exists")
            else:
                raise
        
        # Add subscription_end_date column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_end_date DATETIME')
            print("✅ Added subscription_end_date column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ subscription_end_date column already exists")
            else:
                raise
        
        # Add daily_usage_count column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN daily_usage_count INTEGER DEFAULT 0')
            print("✅ Added daily_usage_count column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ daily_usage_count column already exists")
            else:
                raise
        
        # Add daily_usage_reset_date column
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN daily_usage_reset_date DATETIME')
            print("✅ Added daily_usage_reset_date column")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("⚠️ daily_usage_reset_date column already exists")
            else:
                raise
        
        # Commit all changes
        conn.commit()
        print("✅ Migration completed successfully!")
        
        # Verify the new schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("\n📋 Updated users table schema:")
        for column in columns:
            print(f"  - {column[1]} ({column[2]})")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_users_table()
