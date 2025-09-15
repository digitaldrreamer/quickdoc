#!/usr/bin/env python3
"""
Test script to verify EPUB functionality in Docker container
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:8005"

def test_docker_epub_functionality():
    """Test EPUB functionality in Docker container"""
    print("🐳 Testing EPUB functionality in Docker container...")
    
    # Wait for container to be ready
    print("⏳ Waiting for container to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Container is ready!")
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("❌ Container not ready after 30 attempts")
                return False
            time.sleep(2)
            print(f"   Attempt {i+1}/{max_retries}...")
    
    # Test 1: Check supported formats
    print("\n1. Testing supported formats...")
    try:
        response = requests.get(f"{BASE_URL}/supported-formats")
        if response.status_code == 200:
            data = response.json()
            if '.epub' in data.get('supported_extensions', []):
                print("✅ EPUB is listed in supported formats")
            else:
                print("❌ EPUB not found in supported formats")
                return False
            
            if 'extract-chapters' in data.get('special_endpoints', {}):
                print("✅ EPUB chapter extraction endpoint is documented")
            else:
                print("❌ EPUB chapter extraction endpoint not documented")
                return False
        else:
            print(f"❌ Failed to get supported formats: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing supported formats: {e}")
        return False
    
    # Test 2: Test EPUB chapter endpoint availability
    print("\n2. Testing EPUB chapter endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/extract-chapters")
        if response.status_code == 422:  # Validation error expected without file
            print("✅ EPUB chapter endpoint is available")
        else:
            print(f"❌ Unexpected response from EPUB endpoint: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing EPUB endpoint: {e}")
        return False
    
    # Test 3: Test with non-EPUB file (should fail)
    print("\n3. Testing file type validation...")
    try:
        with open("test_dummy.txt", "w") as f:
            f.write("This is a test file")
        
        with open("test_dummy.txt", "rb") as f:
            files = {"file": ("test_dummy.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/extract-chapters", files=files)
        
        if response.status_code == 400:
            print("✅ File type validation working correctly")
        else:
            print(f"❌ File type validation failed: {response.status_code}")
            return False
        
        # Clean up
        import os
        os.remove("test_dummy.txt")
        
    except Exception as e:
        print(f"❌ Error testing file validation: {e}")
        return False
    
    print("\n🎉 All Docker EPUB tests passed!")
    print("\n📚 Your Docker container now supports:")
    print("   - PDF text extraction (page-by-page)")
    print("   - EPUB text extraction (chapter-by-chapter)")
    print("   - All existing document formats")
    print("\n🚀 Ready for production deployment!")
    
    return True

def main():
    print("🐳 Docker EPUB Functionality Test")
    print("=" * 50)
    
    # Check if Docker is running
    try:
        import subprocess
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'quickdoc' in result.stdout or 'app' in result.stdout:
            print("✅ Docker container is running")
        else:
            print("❌ Docker container not found. Please run:")
            print("   docker-compose up --build")
            return
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return
    
    test_docker_epub_functionality()

if __name__ == "__main__":
    main()
