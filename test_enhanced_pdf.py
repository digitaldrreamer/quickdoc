#!/usr/bin/env python3
"""
Enhanced test script for PDF conversion with new features
"""

import requests
import json
import os
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8001"

# Test markdown with various features
ENHANCED_MARKDOWN = """
# Enhanced Test Document

This document tests the enhanced PDF conversion features.

## Basic Formatting

**Bold text** and *italic text* work as expected.

### Lists

1. First item
2. Second item
   - Nested item
   - Another nested item

### Tables

| Feature | Status | Notes |
|---------|--------|--------|
| Mistune | ✅ | Enhanced with plugins |
| Pandoc | ✅ | Better configuration |
| WeasyPrint | ✅ | More options |

### Code Blocks

```python
def enhanced_converter():
    return "Much better!"
```

### Math (if supported)

The formula $E = mc^2$ is famous.

Block math:
$$
\\int_0^1 x^2 dx = \\frac{1}{3}
$$

### Footnotes

This has a footnote[^1].

[^1]: This is the footnote text.

### Task Lists

- [x] Implement enhanced features
- [ ] Add more tests
- [x] Document new functionality

### Strikethrough

~~This text is struck through~~

### Superscript and Subscript

H~2~O is water, and 2^10^ is 1024.

### Blockquotes

> This is a blockquote with **bold** text.
> 
> It can span multiple lines.

## Advanced Features

This document tests various advanced features of the enhanced PDF converter.
"""

def test_basic_pdf_conversion():
    """Test basic PDF conversion with default settings"""
    print("🔬 Testing basic PDF conversion...")
    
    payload = {
        "content": ENHANCED_MARKDOWN,
        "content_type": "markdown",
        "priority": "auto"
    }
    
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json=payload)
    
    if response.status_code == 200:
        print("✅ Basic conversion successful")
        print(f"   PDF size: {len(response.content)} bytes")
        
        # Save the PDF
        with open("test_basic.pdf", "wb") as f:
            f.write(response.content)
        print("   Saved as: test_basic.pdf")
        
        # Check headers
        if "X-Conversion-Stats" in response.headers:
            stats = json.loads(response.headers["X-Conversion-Stats"])
            print(f"   Conversion time: {stats.get('conversion_time_seconds', 'N/A')}s")
            print(f"   Converter used: {stats.get('converter_used', 'N/A')}")
        
        return True
    else:
        print(f"❌ Basic conversion failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_enhanced_pdf_conversion():
    """Test enhanced PDF conversion with custom settings"""
    print("\n🔬 Testing enhanced PDF conversion...")
    
    payload = {
        "content": ENHANCED_MARKDOWN,
        "content_type": "markdown",
        "priority": "quality",
        "page_size": "A4",
        "page_orientation": "portrait",
        "page_margins": {
            "top": "1in",
            "right": "0.75in",
            "bottom": "1in",
            "left": "0.75in"
        },
        "font_family": "Arial, sans-serif",
        "font_size": "11pt",
        "pdf_title": "Enhanced Test Document",
        "pdf_author": "Test Suite",
        "pdf_subject": "PDF Enhancement Testing",
        "pdf_keywords": "test, pdf, enhancement",
        "generate_toc": True,
        "toc_depth": 2,
        "optimize_images": True,
        "css_content": """
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        h2 { color: #34495e; }
        .highlight { background-color: #fff2cd; }
        """
    }
    
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json=payload)
    
    if response.status_code == 200:
        print("✅ Enhanced conversion successful")
        print(f"   PDF size: {len(response.content)} bytes")
        
        # Save the PDF
        with open("test_enhanced.pdf", "wb") as f:
            f.write(response.content)
        print("   Saved as: test_enhanced.pdf")
        
        return True
    else:
        print(f"❌ Enhanced conversion failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_html_conversion():
    """Test HTML to PDF conversion"""
    print("\n🔬 Testing HTML to PDF conversion...")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HTML Test</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .highlight { background-color: #ffff99; }
        </style>
    </head>
    <body>
        <h1>HTML Test Document</h1>
        <p>This is a <strong>HTML document</strong> with <em>formatting</em>.</p>
        <p class="highlight">This paragraph is highlighted.</p>
        <table border="1">
            <tr><th>Column 1</th><th>Column 2</th></tr>
            <tr><td>Data 1</td><td>Data 2</td></tr>
        </table>
    </body>
    </html>
    """
    
    payload = {
        "content": html_content,
        "content_type": "html",
        "priority": "speed",
        "pdf_title": "HTML Test Document",
        "pdf_author": "Test Suite"
    }
    
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json=payload)
    
    if response.status_code == 200:
        print("✅ HTML conversion successful")
        print(f"   PDF size: {len(response.content)} bytes")
        
        # Save the PDF
        with open("test_html.pdf", "wb") as f:
            f.write(response.content)
        print("   Saved as: test_html.pdf")
        
        return True
    else:
        print(f"❌ HTML conversion failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_pdf_settings():
    """Test PDF settings endpoints"""
    print("\n🔬 Testing PDF settings endpoints...")
    
    # Test GET settings
    response = requests.get(f"{BASE_URL}/pdf-settings")
    if response.status_code == 200:
        print("✅ GET PDF settings successful")
        settings = response.json()
        print(f"   Default settings: {json.dumps(settings, indent=2)}")
    else:
        print(f"❌ GET PDF settings failed: {response.status_code}")
        return False
    
    # Test POST settings
    new_settings = {
        "default_page_size": "Letter",
        "default_font_family": "Times New Roman",
        "default_font_size": "14pt",
        "default_margins": {
            "top": "1.5in",
            "right": "1in",
            "bottom": "1.5in",
            "left": "1in"
        },
        "enable_toc": True,
        "image_optimization": True,
        "pdf_version": "1.7"
    }
    
    response = requests.post(f"{BASE_URL}/pdf-settings", json=new_settings)
    if response.status_code == 200:
        print("✅ POST PDF settings successful")
        result = response.json()
        print(f"   Settings saved: {result['status']}")
        return True
    else:
        print(f"❌ POST PDF settings failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_performance_comparison():
    """Test performance of different converters"""
    print("\n🔬 Testing performance comparison...")
    
    results = {}
    
    def run_perf_test(priority: str) -> bool:
        print(f"   Testing {priority.title()} Priority...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/convert-to-pdf",
            json={"content": ENHANCED_MARKDOWN, "content_type": "markdown", "priority": priority}
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            stats = json.loads(response.headers["X-Conversion-Stats"])
            results[priority] = {
                "request_time": f"{request_time:.2f}s",
                "pdf_size": len(response.content),
                "converter": stats.get("converter_used", "N/A"),
                "conv_time": f"{stats.get('conversion_time_seconds', 0):.2f}s"
            }
            print(f"     ✅ {priority.title()} Priority: {results[priority]['request_time']}, {results[priority]['pdf_size']} bytes")
            return True
        else:
            print(f"     ❌ {priority.title()} Priority failed: {response.status_code}")
            return False

    # Run tests for all priorities
    speed_ok = run_perf_test("speed")
    quality_ok = run_perf_test("quality")
    auto_ok = run_perf_test("auto")
            
    # Print summary table
    print("\n📊 Performance Comparison:")
    print("   Setting          | Request Time | PDF Size | Converter | Conv Time")
    print("   -----------------|--------------|----------|-----------|----------")
    
    for priority, data in results.items():
        print(
            f"   {priority.title()} Priority".ljust(18) +
            f"| {data['request_time']}".ljust(15) +
            f"| {data['pdf_size']}".ljust(11) +
            f"| {data['converter']}".ljust(12) +
            f"| {data['conv_time']}"
        )
        
    return speed_ok and quality_ok and auto_ok

def test_error_handling():
    """Test error handling for various scenarios"""
    print("\n🔬 Testing error handling...")
    
    all_passed = True
    
    # Test empty content
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "",
        "content_type": "markdown"
    })
    if response.status_code == 400:
        print("✅ Empty content error handled correctly")
    else:
        print(f"❌ Empty content error not handled: {response.status_code}")
        all_passed = False
        
    # Test invalid content type
    response = requests.post(f"{BASE_URL}/convert-to-pdf", json={
        "content": "# Hello",
        "content_type": "invalid_type"
    })
    if response.status_code == 422: # Unprocessable Entity for validation errors
        print("✅ Invalid content type error handled correctly")
    else:
        print(f"❌ Invalid content type error not handled: {response.status_code}")
        all_passed = False
        
    return all_passed

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced PDF Conversion Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not running or unhealthy")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return
    
    print("✅ Server is running")
    
    # Run tests
    test_results = []
    
    tests = [
        ("Basic PDF Conversion", test_basic_pdf_conversion),
        ("Enhanced PDF Conversion", test_enhanced_pdf_conversion),
        ("HTML Conversion", test_html_conversion),
        ("PDF Settings", test_pdf_settings),
        ("Performance Comparison", test_performance_comparison),
        ("Error Handling", test_error_handling)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main() 