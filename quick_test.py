#!/usr/bin/env python3

import requests
import json
import time

def test_pdf_conversion():
    """Quick test of PDF conversion endpoint"""
    
    # Simple test content
    test_content = """
# Test Document

This is a **test document** with some basic formatting.

## Features
- Item 1
- Item 2

> This is a blockquote
"""
    
    # Test data
    test_data = {
        "content": test_content,
        "content_type": "markdown",
        "priority": "auto"
    }
    
    try:
        print("Testing PDF conversion endpoint...")
        response = requests.post(
            "http://localhost:8001/convert-to-pdf",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ PDF conversion successful!")
            print(f"  - Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"  - PDF Size: {len(response.content)} bytes")
            print(f"  - Converter Used: {response.headers.get('X-Converter-Used', 'unknown')}")
            print(f"  - Processing Time: {response.headers.get('X-Processing-Time', 'unknown')}s")
            print(f"  - Complexity Detected: {response.headers.get('X-Complexity-Detected', 'unknown')}")
            
            # Save the PDF
            with open("test_output.pdf", "wb") as f:
                f.write(response.content)
            print("  - PDF saved as 'test_output.pdf'")
            
        else:
            print(f"✗ PDF conversion failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Please make sure the server is running on port 8001.")
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_pdf_conversion() 