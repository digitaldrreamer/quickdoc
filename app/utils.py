import os
import magic
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.odt', '.rtf', '.md', '.markdown',
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
}

# MIME type mappings
MIME_TYPE_MAPPINGS = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.oasis.opendocument.text': '.odt',
    'application/rtf': '.rtf',
    'text/markdown': '.md',
    'text/plain': '.txt',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
}

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension in lowercase
    """
    return Path(filename).suffix.lower()

def is_supported_file_type(filename: str) -> bool:
    """
    Check if file type is supported based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if supported, False otherwise
    """
    extension = get_file_extension(filename)
    return extension in SUPPORTED_EXTENSIONS

def get_file_type(file_path: str) -> Optional[str]:
    """
    Detect file type using python-magic.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected file extension or None if unknown
    """
    try:
        mime_type = magic.from_file(file_path, mime=True)
        return MIME_TYPE_MAPPINGS.get(mime_type)
    except Exception as e:
        logger.warning(f"Could not detect file type for {file_path}: {e}")
        return None

def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate file size.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if file size is within limits
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    except Exception as e:
        logger.error(f"Error checking file size for {file_path}: {e}")
        return False

def is_pdf_file(file_path: str) -> bool:
    """
    Check if file is a PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is PDF
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an image
    """
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)
    
    # Join lines with single newlines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive newlines
    import re
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def format_file_info(filename: str, file_size: int, file_type: str) -> dict:
    """
    Format file information for logging/response.
    
    Args:
        filename: Name of the file
        file_size: Size in bytes
        file_type: File type/extension
        
    Returns:
        Formatted file info dict
    """
    return {
        'filename': filename,
        'file_size_bytes': file_size,
        'file_size_mb': round(file_size / (1024 * 1024), 2),
        'file_type': file_type,
        'is_supported': is_supported_file_type(filename)
    }
