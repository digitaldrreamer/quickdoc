#!/usr/bin/env python3
"""
CLI script for testing document text extraction without hitting the API.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.processor import extract_text_from_doc
from app.utils import is_supported_file_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Extract text from documents')
    parser.add_argument('file_path', help='Path to the document file')
    parser.add_argument('--output', '-o', help='Output file path (default: stdout)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (no logging)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Check if file type is supported
    if not is_supported_file_type(args.file_path):
        print(f"Error: File type not supported", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Extract text
        filename = os.path.basename(args.file_path)
        text = await extract_text_from_doc(args.file_path, filename)
        
        # Output result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text extracted and saved to '{args.output}'")
        else:
            print("=" * 50)
            print("EXTRACTED TEXT:")
            print("=" * 50)
            print(text)
            print("=" * 50)
            print(f"Character count: {len(text)}")
    
    except Exception as e:
        print(f"Error extracting text: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 