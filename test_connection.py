#!/usr/bin/env python3
"""
Test script to verify database connection works correctly.
"""

import os
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine, text

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded .env from: {env_file}")
except ImportError:
    print("Warning: python-dotenv not available")

# Get connection parameters
db_user = os.getenv("SUPABASE_DB_USER")
db_password = os.getenv("SUPABASE_DB_PASSWORD")
db_host = os.getenv("SUPABASE_DB_HOST", "").replace('http://', '').replace('https://', '')
db_port = os.getenv("SUPABASE_DB_PORT", "5432")
db_name = os.getenv("SUPABASE_DB_NAME")

print("\nConnection Parameters:")
print(f"  User: {db_user}")
print(f"  Host: {db_host}")
print(f"  Port: {db_port}")
print(f"  Database: {db_name}")
print(f"  Password: {'SET' if db_password else 'NOT SET'}")

# Validate parameters
if not all([db_user, db_password, db_host, db_name]):
    print("\n✗ Missing required connection parameters!")
    exit(1)

# Test 1: Direct psycopg2 connection
print("\n1. Testing direct psycopg2 connection...")
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        connect_timeout=10
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"✓ psycopg2 connection successful!")
    print(f"  PostgreSQL version: {version[:50]}...")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"✗ psycopg2 connection failed: {e}")

# Test 2: SQLAlchemy connection
print("\n2. Testing SQLAlchemy connection...")
try:
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url, pool_timeout=10, connect_args={"connect_timeout": 10})
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database(), current_user;"))
        db_info = result.fetchone()
        print(f"✓ SQLAlchemy connection successful!")
        print(f"  Current database: {db_info[0]}")
        print(f"  Current user: {db_info[1]}")
    
    engine.dispose()
    
except Exception as e:
    print(f"✗ SQLAlchemy connection failed: {e}")

# Test 3: Check if required tables exist
print("\n3. Testing table access...")
try:
    engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('floats', 'profiles', 'float_embeddings')
        """))
        
        existing_tables = [row[0] for row in result.fetchall()]
        
        expected_tables = ['floats', 'profiles', 'float_embeddings']
        
        print(f"  Expected tables: {expected_tables}")
        print(f"  Existing tables: {existing_tables}")
        
        if set(expected_tables).issubset(set(existing_tables)):
            print("✓ All required tables exist!")
        else:
            missing = set(expected_tables) - set(existing_tables)
            print(f"⚠ Missing tables: {missing}")
            print("  Run 'python main.py --setup-db' to create them")
    
    engine.dispose()
    
except Exception as e:
    print(f"✗ Table check failed: {e}")

print("\nConnection test completed!")
print("If all tests passed, you can run: python main.py --setup-db")