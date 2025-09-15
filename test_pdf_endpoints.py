#!/usr/bin/env python3
"""
Test script for PDF conversion endpoints
"""

import requests
import json
import os
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8005"
TEST_MARKDOWN = """
# Test Document

This is a **test document** with some formatting.

## Features

- Simple list items
- *Italic text*
- `Code snippets`

### Table Example

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

### Code Block

```python
def hello_world():
    print("Hello, World!")
```

> This is a blockquote

## Complex Features (should trigger pandoc)

This document contains math: $E = mc^2$

And footnotes[^1].

[^1]: This is a footnote.
"""

TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Test HTML Document</h1>
    <p>This is a <strong>test HTML document</strong> with some formatting.</p>
    
    <h2>Features</h2>
    <ul>
        <li>Simple list items</li>
        <li><em>Italic text</em></li>
        <li><code>Code snippets</code></li>
    </ul>
    
    <h3>Table Example</h3>
    <table>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
        </tr>
        <tr>
            <td>Data 1</td>
            <td>Data 2</td>
        </tr>
        <tr>
            <td>Data 3</td>
            <td>Data 4</td>
        </tr>
    </table>
    
    <blockquote>
        This is a blockquote
    </blockquote>
</body>
</html>
"""

def test_pdf_conversion_endpoint():
    """Test the /convert-to-pdf endpoint"""
    print("Testing /convert-to-pdf endpoint...")
    
    # Test 1: Simple Markdown with auto priority
    print("\n1. Testing simple markdown with auto priority...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "# Simple Test\n\nThis is a simple markdown document.",
        "content_type": "markdown",
        "priority": "auto"
    })
    
    if response.status_code == 200:
        print("✓ Simple markdown conversion successful")
        print(f"  - Converter used: {response.headers.get('X-Converter-Used', 'Unknown')}")
        print(f"  - Processing time: {response.headers.get('X-Processing-Time', 'Unknown')}s")
        print(f"  - PDF size: {response.headers.get('X-PDF-Size', 'Unknown')} bytes")
        print(f"  - Complexity detected: {response.headers.get('X-Complexity-Detected', 'Unknown')}")
        
        # Save PDF
        with open("test_simple.pdf", "wb") as f:
            f.write(response.content)
    else:
        print(f"✗ Simple markdown conversion failed: {response.status_code}")
        print(f"  Error: {response.text}")
    
    # Test 2: Complex Markdown with auto priority
    print("\n2. Testing complex markdown with auto priority...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": TEST_MARKDOWN,
        "content_type": "markdown",
        "priority": "auto"
    })
    
    if response.status_code == 200:
        print("✓ Complex markdown conversion successful")
        print(f"  - Converter used: {response.headers.get('X-Converter-Used', 'Unknown')}")
        print(f"  - Processing time: {response.headers.get('X-Processing-Time', 'Unknown')}s")
        print(f"  - PDF size: {response.headers.get('X-PDF-Size', 'Unknown')} bytes")
        print(f"  - Complexity detected: {response.headers.get('X-Complexity-Detected', 'Unknown')}")
        
        # Save PDF
        with open("test_complex.pdf", "wb") as f:
            f.write(response.content)
    else:
        print(f"✗ Complex markdown conversion failed: {response.status_code}")
        print(f"  Error: {response.text}")
    
    # Test 3: HTML with speed priority
    print("\n3. Testing HTML with speed priority...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": TEST_HTML,
        "content_type": "html",
        "priority": "speed"
    })
    
    if response.status_code == 200:
        print("✓ HTML conversion successful")
        print(f"  - Converter used: {response.headers.get('X-Converter-Used', 'Unknown')}")
        print(f"  - Processing time: {response.headers.get('X-Processing-Time', 'Unknown')}s")
        print(f"  - PDF size: {response.headers.get('X-PDF-Size', 'Unknown')} bytes")
        print(f"  - Complexity detected: {response.headers.get('X-Complexity-Detected', 'Unknown')}")
        
        # Save PDF
        with open("test_html.pdf", "wb") as f:
            f.write(response.content)
    else:
        print(f"✗ HTML conversion failed: {response.status_code}")
        print(f"  Error: {response.text}")
    
    # Test 4: Markdown with quality priority
    print("\n4. Testing markdown with quality priority...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": TEST_MARKDOWN,
        "content_type": "markdown",
        "priority": "quality"
    })
    
    if response.status_code == 200:
        print("✓ Quality markdown conversion successful")
        print(f"  - Converter used: {response.headers.get('X-Converter-Used', 'Unknown')}")
        print(f"  - Processing time: {response.headers.get('X-Processing-Time', 'Unknown')}s")
        print(f"  - PDF size: {response.headers.get('X-PDF-Size', 'Unknown')} bytes")
        print(f"  - Complexity detected: {response.headers.get('X-Complexity-Detected', 'Unknown')}")
        
        # Save PDF
        with open("test_quality.pdf", "wb") as f:
            f.write(response.content)
    else:
        print(f"✗ Quality markdown conversion failed: {response.status_code}")
        print(f"  Error: {response.text}")

    # Test 5: With custom CSS
    print("\n5. Testing with custom CSS...")
    custom_css = """
    body {
        font-family: 'Arial', sans-serif;
        color: #333;
        background-color: #f9f9f9;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    """
    
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "# Custom Styled Document\n\nThis document has custom CSS styling.",
        "content_type": "markdown",
        "priority": "auto",
        "css_content": custom_css
    })
    
    if response.status_code == 200:
        print("✓ Custom CSS conversion successful")
        print(f"  - Converter used: {response.headers.get('X-Converter-Used', 'Unknown')}")
        print(f"  - Processing time: {response.headers.get('X-Processing-Time', 'Unknown')}s")
        print(f"  - PDF size: {response.headers.get('X-PDF-Size', 'Unknown')} bytes")
        
        # Save PDF
        with open("test_custom_css.pdf", "wb") as f:
            f.write(response.content)
    else:
        print(f"✗ Custom CSS conversion failed: {response.status_code}")
        print(f"  Error: {response.text}")

def test_error_handling():
    """Test error handling"""
    print("\n\nTesting error handling...")
    
    # Test 1: Empty content
    print("\n1. Testing empty content...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "",
        "content_type": "markdown",
        "priority": "auto"
    })
    
    if response.status_code == 400:
        print("✓ Empty content properly rejected")
    else:
        print(f"✗ Empty content not handled properly: {response.status_code}")
    
    # Test 2: Invalid content type
    print("\n2. Testing invalid content type...")
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "Test content",
        "content_type": "invalid",
        "priority": "auto"
    })
    
    if response.status_code == 422:
        print("✓ Invalid content type properly rejected")
    else:
        print(f"✗ Invalid content type not handled properly: {response.status_code}")

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            return True
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server: {e}")
        return False

def main():
    print("PDF Conversion API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_status():
        print("\nPlease start the server first:")
        print("  python -m uvicorn app.main:app --reload")
        return
    
    # Run tests
    test_pdf_conversion_endpoint()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nGenerated test files:")
    test_files = [
        "test_simple.pdf",
        "test_complex.pdf", 
        "test_html.pdf",
        "test_quality.pdf",
        "test_custom_css.pdf"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  - {file} ({size} bytes)")
    
    print("\nYou can open these PDF files to verify the conversion results.")

if __name__ == "__main__":
    main() 