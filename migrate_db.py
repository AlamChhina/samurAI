#!/usr/bin/env python3
"""
Database migration script to add summary-related and media_hash columns to existing jobs table.
This script is idempotent - it can be run multiple times safely.
"""

import sqlite3
import os
import hashlib
from pathlib import Path

DATABASE_PATH = "video_audio_service.db"

def migrate_database():
    """Add summary-related columns and media_hash to jobs table if they don't exist"""
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
        if 'media_hash' not in columns:
            columns_to_add.append(('media_hash', 'TEXT'))
        
        # Add missing columns
        for column_name, column_type in columns_to_add:
            print(f"Adding column: {column_name} {column_type}")
            cursor.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}")
        
        # Create index on media_hash if it was just added
        if 'media_hash' in [col[0] for col in columns_to_add]:
            print("Creating index on media_hash column...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_media_hash ON jobs(media_hash)")
        
        if columns_to_add:
            conn.commit()
            print(f"Successfully added {len(columns_to_add)} columns to jobs table")
        else:
            print("No migration needed - all columns already exist")
            
        # Populate media_hash for existing records
        populate_missing_media_hashes(cursor, conn)
            
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

def populate_missing_media_hashes(cursor, conn):
    """Populate media_hash for existing records that don't have it"""
    
    print("🔄 Checking for records without media_hash...")
    
    # Find jobs without media_hash
    cursor.execute("SELECT id, content_hash FROM jobs WHERE media_hash IS NULL OR media_hash = ''")
    jobs_without_media_hash = cursor.fetchall()
    
    if not jobs_without_media_hash:
        print("✅ All jobs already have media_hash values")
        return
    
    print(f"📝 Found {len(jobs_without_media_hash)} jobs without media_hash")
    
    updated_count = 0
    for job_id, content_hash in jobs_without_media_hash:
        try:
            if content_hash:
                # Generate media_hash from content_hash (remove the prompt part)
                # Format was like "video:filehash:prompt" or "audio:filehash:prompt" or "youtube:videoid:prompt"
                content_parts = content_hash.split(':')
                if len(content_parts) >= 2:
                    # Reconstruct media identifier without the prompt
                    media_type = content_parts[0]  # video, audio, or youtube
                    media_id = content_parts[1]    # file hash or video ID
                    media_identifier = f"{media_type}:{media_id}"
                    
                    # Generate media hash
                    media_hash = hashlib.sha256(media_identifier.encode('utf-8')).hexdigest()
                    
                    # Update the record
                    cursor.execute("UPDATE jobs SET media_hash = ? WHERE id = ?", (media_hash, job_id))
                    updated_count += 1
                    
                    if updated_count <= 5:  # Show first few for verification
                        print(f"   Job {job_id}: {media_identifier} -> {media_hash[:16]}...")
                
        except Exception as e:
            print(f"   Warning: Could not generate media_hash for job {job_id}: {e}")
    
    # Commit all changes
    conn.commit()
    print(f"✅ Updated {updated_count} jobs with media_hash values")

if __name__ == "__main__":
    print("=== Database Migration ===")
    print()
    migrate_database()
    print()
    print("🎉 Database migration completed!")
