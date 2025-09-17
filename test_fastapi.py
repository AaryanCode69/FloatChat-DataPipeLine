"""
Test script for FloatChat FastAPI application
============================================

This script tests the FastAPI endpoints to ensure they work correctly.
"""

import requests
import time
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8001"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed: {health_data['status']}")
            print(f"  Services: {health_data['services']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            root_data = response.json()
            print(f"✓ Root endpoint working: {root_data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_file_upload():
    """Test file upload with a sample NetCDF file."""
    print("🔍 Testing file upload...")
    
    # Check if we have any sample NetCDF files
    sample_files = []
    data_dirs = [
        Path("2019"),
        Path("argo_data"),
        Path("argo_data_2020_01"),
        Path("data")
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            nc_files = list(data_dir.glob("*.nc"))
            if nc_files:
                sample_files.extend(nc_files[:1])  # Take one file from each directory
    
    if not sample_files:
        print("⚠️  No sample NetCDF files found for testing")
        return False
    
    # Test with the first available file
    test_file = sample_files[0]
    print(f"📁 Using test file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'application/octet-stream')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            upload_result = response.json()
            task_id = upload_result['task_id']
            print(f"✓ File uploaded successfully, task ID: {task_id}")
            
            # Wait a bit and check status
            print("⏳ Waiting for processing...")
            time.sleep(5)
            
            status_response = requests.get(f"{BASE_URL}/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 Processing status: {status_data['status']}")
                print(f"💬 Message: {status_data['message']}")
                
                if status_data['status'] == 'completed':
                    print("✅ File processing completed successfully!")
                    if 'extracted_data' in status_data:
                        print("📋 Extracted data preview:")
                        extracted = status_data['extracted_data']
                        print(f"  - Profiles: {extracted.get('total_profiles', 'N/A')}")
                        print(f"  - Measurements: {list(extracted.get('measurements', {}).keys())}")
                    
                    if 'storage_results' in status_data:
                        storage = status_data['storage_results']
                        print("💾 Storage results:")
                        print(f"  - ChromaDB: {storage.get('chromadb', {}).get('status', 'N/A')}")
                        print(f"  - Supabase: {storage.get('supabase', {}).get('status', 'N/A')}")
                
                return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return False
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("🚀 Starting FastAPI tests...")
    print("=" * 50)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_endpoint),
        ("File Upload", test_file_upload)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        results[test_name] = test_func()
        print("-" * 30)
    
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI application is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    print("FastAPI Test Suite")
    print("Make sure the FastAPI server is running on localhost:8001")
    print("You can start it with: python fastapi_app.py")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            run_tests()
        else:
            print("❌ FastAPI server not responding correctly")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI server. Make sure it's running on localhost:8001")
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")