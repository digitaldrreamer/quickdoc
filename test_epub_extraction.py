#!/usr/bin/env python3
"""
Test script for EPUB text extraction endpoints
"""

import requests
import json
import os
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"

def test_epub_extraction_endpoint():
    """Test the /extract-chapters endpoint"""
    print("Testing /extract-chapters endpoint...")
    
    # Test 1: Check if endpoint exists
    print("\n1. Testing endpoint availability...")
    try:
        # Try to make a request without a file to check if endpoint exists
        response = requests.post(f"{BASE_URL}/extract-chapters")
        if response.status_code == 422:  # Validation error expected without file
            print("✓ Endpoint exists and responds correctly")
        else:
            print(f"✗ Unexpected response: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server")
        return False
    except Exception as e:
        print(f"✗ Error testing endpoint: {e}")
        return False
    
    # Test 2: Test with non-EPUB file (should fail)
    print("\n2. Testing with non-EPUB file...")
    try:
        # Create a dummy text file
        with open("test_dummy.txt", "w") as f:
            f.write("This is a test file")
        
        with open("test_dummy.txt", "rb") as f:
            files = {"file": ("test_dummy.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/extract-chapters", files=files)
        
        if response.status_code == 400:
            print("✓ Non-EPUB file properly rejected")
        else:
            print(f"✗ Non-EPUB file not handled properly: {response.status_code}")
            print(f"  Response: {response.text}")
        
        # Clean up
        os.remove("test_dummy.txt")
        
    except Exception as e:
        print(f"✗ Error testing non-EPUB file: {e}")
    
    # Test 3: Test with empty file
    print("\n3. Testing with empty file...")
    try:
        with open("test_empty.epub", "wb") as f:
            f.write(b"")
        
        with open("test_empty.epub", "rb") as f:
            files = {"file": ("test_empty.epub", f, "application/epub+zip")}
            response = requests.post(f"{BASE_URL}/extract-chapters", files=files)
        
        if response.status_code == 400:
            print("✓ Empty file properly rejected")
        else:
            print(f"✗ Empty file not handled properly: {response.status_code}")
        
        # Clean up
        os.remove("test_empty.epub")
        
    except Exception as e:
        print(f"✗ Error testing empty file: {e}")
    
    print("\n✅ Basic endpoint tests completed!")
    print("\nTo test with a real EPUB file:")
    print("1. Place an EPUB file in this directory")
    print("2. Run: python test_epub_extraction.py <epub_filename>")
    print("3. Or use curl:")
    print("   curl -X POST -F 'file=@your_file.epub' http://localhost:8005/extract-chapters")
    
    return True

def test_with_real_epub(epub_filename: str):
    """Test with a real EPUB file"""
    print(f"\n🔬 Testing with real EPUB: {epub_filename}")
    
    if not os.path.exists(epub_filename):
        print(f"✗ EPUB file not found: {epub_filename}")
        return False
    
    try:
        with open(epub_filename, "rb") as f:
            files = {"file": (epub_filename, f, "application/epub+zip")}
            response = requests.post(f"{BASE_URL}/extract-chapters", files=files)
        
        if response.status_code == 200:
            print("✅ EPUB processing successful!")
            
            # Parse response
            data = response.json()
            
            print(f"   📚 Chapters extracted: {data['chapter_count']}")
            print(f"   📁 Filename: {data['filename']}")
            print(f"   📊 File type: {data['file_type']}")
            
            # Show metrics
            metrics = data.get('metrics', {})
            if metrics:
                print(f"   ⏱️  Processing time: {metrics.get('processing_time_seconds', 'N/A')}s")
                print(f"   🔧 Method used: {metrics.get('method_used', 'N/A')}")
                print(f"   📈 Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
            
            # Show first few chapters
            chapters = data.get('chapters', [])
            print(f"\n   📖 First few chapters preview:")
            for i, chapter_text in enumerate(chapters[:3]):  # Show first 3 chapters
                preview = chapter_text[:100] + "..." if len(chapter_text) > 100 else chapter_text
                print(f"      Chapter {i+1}: {preview}")
            
            if len(chapters) > 3:
                print(f"      ... and {len(chapters) - 3} more chapters")
            
            # Save results to file
            output_file = f"{epub_filename}_chapters.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n   💾 Results saved to: {output_file}")
            
            return True
            
        else:
            print(f"❌ EPUB processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing EPUB: {e}")
        return False

def test_regular_extract_with_epub(epub_filename: str):
    """Test the regular /extract endpoint with EPUB file"""
    print(f"\n🔬 Testing regular /extract endpoint with EPUB: {epub_filename}")
    
    if not os.path.exists(epub_filename):
        print(f"✗ EPUB file not found: {epub_filename}")
        return False
    
    try:
        with open(epub_filename, "rb") as f:
            files = {"file": (epub_filename, f, "application/epub+zip")}
            response = requests.post(f"{BASE_URL}/extract", files=files)
        
        if response.status_code == 200:
            print("✅ Regular EPUB extraction successful!")
            
            # Parse response
            data = response.json()
            
            print(f"   📁 Filename: {data['filename']}")
            print(f"   📊 File type: {data['file_type']}")
            print(f"   📝 Character count: {data['character_count']}")
            
            # Show metrics
            metrics = data.get('metrics', {})
            if metrics:
                print(f"   ⏱️  Processing time: {metrics.get('processing_time_seconds', 'N/A')}s")
                print(f"   🔧 Method used: {metrics.get('method_used', 'N/A')}")
            
            # Show text preview
            text = data.get('text', '')
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n   📖 Text preview: {preview}")
            
            return True
            
        else:
            print(f"❌ Regular EPUB extraction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing EPUB: {e}")
        return False

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def main():
    print("🚀 EPUB Text Extraction Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_status():
        print("\nPlease start the server first:")
        print("  python -m uvicorn app.main:app --reload")
        return
    
    # Run basic tests
    test_epub_extraction_endpoint()
    
    # Test with real EPUB if provided
    import sys
    if len(sys.argv) > 1:
        epub_file = sys.argv[1]
        test_with_real_epub(epub_file)
        test_regular_extract_with_epub(epub_file)
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\nAPI Endpoints:")
    print("  POST /extract-chapters - Extract EPUB chapters separately")
    print("  POST /extract - Extract EPUB as single text")
    print("\nResponse format for /extract-chapters:")
    print("""
{
  "chapters": ["chapter 1 text", "chapter 2 text", ...],
  "chapter_count": 2,
  "filename": "book.epub",
  "file_type": ".epub",
  "metrics": {
    "processing_time_seconds": 1.23,
    "method_used": "epub_chapter_extraction",
    "memory_usage_mb": 45.6
  }
}
""")

if __name__ == "__main__":
    main()
