#!/usr/bin/env python3
"""
Test script for page-by-page PDF text extraction endpoint
"""

import requests
import json
import os
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8005"

def test_page_extraction_endpoint():
    """Test the /extract-pages endpoint"""
    print("Testing /extract-pages endpoint...")
    
    # Test 1: Check if endpoint exists
    print("\n1. Testing endpoint availability...")
    try:
        # Try to make a request without a file to check if endpoint exists
        response = requests.post(f"{BASE_URL}/extract-pages")
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
    
    # Test 2: Test with non-PDF file (should fail)
    print("\n2. Testing with non-PDF file...")
    try:
        # Create a dummy text file
        with open("test_dummy.txt", "w") as f:
            f.write("This is a test file")
        
        with open("test_dummy.txt", "rb") as f:
            files = {"file": ("test_dummy.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/extract-pages", files=files)
        
        if response.status_code == 400:
            print("✓ Non-PDF file properly rejected")
        else:
            print(f"✗ Non-PDF file not handled properly: {response.status_code}")
            print(f"  Response: {response.text}")
        
        # Clean up
        os.remove("test_dummy.txt")
        
    except Exception as e:
        print(f"✗ Error testing non-PDF file: {e}")
    
    # Test 3: Test with empty file
    print("\n3. Testing with empty file...")
    try:
        with open("test_empty.pdf", "wb") as f:
            f.write(b"")
        
        with open("test_empty.pdf", "rb") as f:
            files = {"file": ("test_empty.pdf", f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/extract-pages", files=files)
        
        if response.status_code == 400:
            print("✓ Empty file properly rejected")
        else:
            print(f"✗ Empty file not handled properly: {response.status_code}")
        
        # Clean up
        os.remove("test_empty.pdf")
        
    except Exception as e:
        print(f"✗ Error testing empty file: {e}")
    
    print("\n✅ Basic endpoint tests completed!")
    print("\nTo test with a real PDF file:")
    print("1. Place a PDF file in this directory")
    print("2. Run: python test_page_extraction.py <pdf_filename>")
    print("3. Or use curl:")
    print("   curl -X POST -F 'file=@your_file.pdf' http://localhost:8005/extract-pages")
    
    return True

def test_with_real_pdf(pdf_filename: str):
    """Test with a real PDF file"""
    print(f"\n🔬 Testing with real PDF: {pdf_filename}")
    
    if not os.path.exists(pdf_filename):
        print(f"✗ PDF file not found: {pdf_filename}")
        return False
    
    try:
        with open(pdf_filename, "rb") as f:
            files = {"file": (pdf_filename, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/extract-pages", files=files)
        
        if response.status_code == 200:
            print("✅ PDF processing successful!")
            
            # Parse response
            data = response.json()
            
            print(f"   📄 Pages extracted: {data['page_count']}")
            print(f"   📁 Filename: {data['filename']}")
            print(f"   📊 File type: {data['file_type']}")
            
            # Show metrics
            metrics = data.get('metrics', {})
            if metrics:
                print(f"   ⏱️  Processing time: {metrics.get('processing_time_seconds', 'N/A')}s")
                print(f"   🔧 Method used: {metrics.get('method_used', 'N/A')}")
                print(f"   📈 Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
            
            # Show first few pages
            pages = data.get('pages', [])
            print(f"\n   📖 First few pages preview:")
            for i, page_text in enumerate(pages[:3]):  # Show first 3 pages
                preview = page_text[:100] + "..." if len(page_text) > 100 else page_text
                print(f"      Page {i+1}: {preview}")
            
            if len(pages) > 3:
                print(f"      ... and {len(pages) - 3} more pages")
            
            # Save results to file
            output_file = f"{pdf_filename}_pages.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n   💾 Results saved to: {output_file}")
            
            return True
            
        else:
            print(f"❌ PDF processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
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
    print("🚀 PDF Page-by-Page Extraction Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_status():
        print("\nPlease start the server first:")
        print("  python -m uvicorn app.main:app --reload")
        return
    
    # Run basic tests
    test_page_extraction_endpoint()
    
    # Test with real PDF if provided
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        test_with_real_pdf(pdf_file)
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\nAPI Endpoint: POST /extract-pages")
    print("Response format:")
    print("""
{
  "pages": ["page 1 text", "page 2 text", ...],
  "page_count": 2,
  "filename": "document.pdf",
  "file_type": ".pdf",
  "metrics": {
    "processing_time_seconds": 1.23,
    "method_used": "pdfminer_pages",
    "memory_usage_mb": 45.6
  }
}
""")

if __name__ == "__main__":
    main()
