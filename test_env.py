#!/usr/bin/env python3
"""
Test script to verify environment variables are loaded correctly.
"""

import os
from pathlib import Path

# Try to load dotenv
try:
    from dotenv import load_dotenv
    print("✓ python-dotenv is available")
except ImportError:
    print("✗ python-dotenv is not available")
    load_dotenv = None

# Load .env file
env_file = Path(__file__).parent / '.env'
print(f"Looking for .env file at: {env_file}")
print(f".env file exists: {env_file.exists()}")

if env_file.exists():
    if load_dotenv:
        load_dotenv(env_file)
        print("✓ Loaded .env using python-dotenv")
    else:
        # Manual loading
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✓ Manually loaded .env file")

# Check environment variables
print("\nEnvironment Variables:")
required_vars = [
    "SUPABASE_DB_USER",
    "SUPABASE_DB_PASSWORD", 
    "SUPABASE_DB_HOST",
    "SUPABASE_DB_PORT",
    "SUPABASE_DB_NAME"
]

all_set = True
for var in required_vars:
    value = os.getenv(var)
    if value:
        if 'PASSWORD' in var:
            print(f"  {var}: ****** (SET)")
        else:
            print(f"  {var}: {value}")
    else:
        print(f"  {var}: NOT SET")
        all_set = False

print(f"\nAll required variables set: {'✓' if all_set else '✗'}")

if all_set:
    print("\n✓ Environment is properly configured!")
    print("You can now run: python main.py --setup-db")
else:
    print("\n✗ Some environment variables are missing.")
    print("Please check your .env file.")