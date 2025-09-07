import os
import subprocess
import tempfile
import logging
from typing import Optional, List
from pathlib import Path
import asyncio
import io

from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO

# Try to import PyMuPDF with proper error handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None  # type: ignore
    logging.warning("PyMuPDF not available, will use alternative PDF processing")

from PIL import Image
import cv2
import numpy as np
from .metrics import ResourceTracker
from .core.config import settings

# Try to import EPUB libraries with proper error handling
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    epub = None  # type: ignore
    BeautifulSoup = None  # type: ignore
    logging.warning("EPUB libraries not available, will skip EPUB processing")

logger = logging.getLogger(__name__)

# Global PaddleOCR instance (lazy loaded)
_ocr_instance = None

def get_ocr_instance():
    """Get or create PaddleOCR instance with detailed logging."""
    if not settings.ENABLE_IMAGE_OCR:
        raise Exception("OCR functionality is disabled. Set ENABLE_IMAGE_OCR=true to enable OCR features.")
    
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR, paddleocr
            
            # Set paddleocr logger to DEBUG to see download progress
            paddleocr.logger.setLevel(logging.DEBUG)

            logger.info("Initializing PaddleOCR... This may take a moment on first run as models are downloaded.")
            
            # Specify a cache directory within the project
            cache_dir = os.path.join(os.path.expanduser("~"), ".paddleocr_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            _ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv3', show_log=True, use_gpu=False, det_model_dir=f'{cache_dir}/det', rec_model_dir=f'{cache_dir}/rec', cls_model_dir=f'{cache_dir}/cls')
            
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
            raise
    return _ocr_instance

async def extract_text_from_doc(file_path: str, filename: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Main function to extract text from various document types.
    
    Args:
        file_path: Path to the temporary file
        filename: Original filename (for extension detection)
        tracker: Optional resource tracker instance
    
    Returns:
        Extracted text as string
    """
    
    file_ext = Path(filename).suffix.lower()
    
    try:
        if file_ext == '.pdf':
            if not settings.ENABLE_PDF_PROCESSING:
                raise ValueError("PDF processing is disabled. Set ENABLE_PDF_PROCESSING=true to enable PDF support.")
            return await extract_text_from_pdf(file_path, tracker)
        elif file_ext in ['.docx', '.odt', '.rtf']:
            if not settings.ENABLE_DOCX_PROCESSING:
                raise ValueError("Document processing (DOCX/ODT/RTF) is disabled. Set ENABLE_DOCX_PROCESSING=true to enable support for these formats.")
            return await extract_text_with_pandoc(file_path, file_ext, tracker)
        elif file_ext in ['.md', '.markdown']:
            if not settings.ENABLE_MARKDOWN_PROCESSING:
                raise ValueError("Markdown processing is disabled. Set ENABLE_MARKDOWN_PROCESSING=true to enable Markdown support.")
            return await extract_text_from_markdown(file_path, tracker)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            if not settings.ENABLE_IMAGE_OCR:
                raise ValueError("Image OCR processing is disabled. Set ENABLE_IMAGE_OCR=true to enable image text extraction.")
            return await extract_text_from_image(file_path, tracker)
        elif file_ext == '.epub':
            if not settings.ENABLE_EPUB_PROCESSING:
                raise ValueError("EPUB processing is disabled. Set ENABLE_EPUB_PROCESSING=true to enable EPUB support.")
            return await extract_text_from_epub(file_path, tracker)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise

async def extract_text_from_pdf(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from PDF using pdfminer.six, fallback to OCR if needed.
    """
    
    try:
        # First try pdfminer.six for text-based PDFs
        text = pdfminer_extract_text(file_path)
        
        if text and len(text.strip()) > 50:  # Reasonable amount of text found
            logger.info("Successfully extracted text from PDF using pdfminer")
            if tracker:
                tracker.log_method("pdfminer")
            return text.strip()
        
        # ! Check if OCR is enabled before attempting OCR
        if not settings.ENABLE_IMAGE_OCR:
            logger.warning("PDF appears to be scanned but OCR is disabled. Returning minimal text extracted.")
            if tracker:
                tracker.log_method("pdfminer_no_ocr")
            return text.strip() if text else ""
        
        logger.info("PDF appears to be scanned or has minimal text, trying OCR")
        
        # If minimal text, try OCR approach
        return await extract_text_from_pdf_ocr(file_path, tracker)
        
    except Exception as e:
        logger.warning(f"pdfminer failed: {e}. Trying OCR approach")
        
        # ! Check if OCR is enabled before attempting OCR fallback
        if not settings.ENABLE_IMAGE_OCR:
            logger.error("PDF processing failed and OCR is disabled. Cannot extract text.")
            raise Exception("PDF processing failed and OCR is disabled. Enable OCR with ENABLE_IMAGE_OCR=true to handle scanned PDFs.")
        
        return await extract_text_from_pdf_ocr(file_path, tracker)

async def extract_text_from_pdf_ocr(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from PDF using OCR (for scanned PDFs).
    """
    if tracker:
        tracker.log_method("pdf_ocr_pymupdf")

    if PYMUPDF_AVAILABLE:
        try:
            # Convert PDF to images using PyMuPDF
            doc = fitz.open(file_path)
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Run OCR on the image
                page_text = await extract_text_from_image_data(img)
                if page_text:
                    full_text.append(page_text)
            
            doc.close()
            
            result = "\n\n".join(full_text)
            logger.info(f"OCR extracted {len(result)} characters from PDF")
            return result
            
        except Exception as e:
            logger.error(f"PyMuPDF OCR extraction failed: {e}")
            # Fallback to pdftoppm
            return await extract_text_from_pdf_pdftoppm(file_path, tracker)
    else:
        # Fallback to pdftoppm if PyMuPDF is not available
        return await extract_text_from_pdf_pdftoppm(file_path, tracker)

async def extract_text_from_pdf_pdftoppm(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from PDF using pdftoppm + OCR as fallback.
    """
    if tracker:
        tracker.log_method("pdf_ocr_pdftoppm")
    
    try:
        # Convert PDF to images using pdftoppm
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                'pdftoppm', 
                '-png', 
                file_path, 
                os.path.join(temp_dir, 'page')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"pdftoppm failed: {result.stderr}")
            
            # Process each generated image
            full_text = []
            image_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
            
            for image_file in image_files:
                image_path = os.path.join(temp_dir, image_file)
                page_text = await extract_text_from_image(image_path)
                if page_text:
                    full_text.append(page_text)
            
            result = "\n\n".join(full_text)
            logger.info(f"pdftoppm + OCR extracted {len(result)} characters")
            return result
            
    except Exception as e:
        logger.error(f"pdftoppm + OCR failed: {e}")
        raise

async def extract_text_with_pandoc(file_path: str, file_ext: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text using pandoc for DOCX, ODT, RTF files.
    """
    if tracker:
        tracker.log_method(f"pandoc_{file_ext.strip('.')}")
    
    try:
        cmd = ['pandoc', file_path, '-t', 'plain']
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            raise Exception(f"pandoc failed: {result.stderr}")
        
        text = result.stdout.strip()
        logger.info(f"pandoc extracted {len(text)} characters from {file_ext}")
        return text
        
    except Exception as e:
        logger.error(f"pandoc extraction failed: {e}")
        raise

async def extract_text_from_markdown(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from Markdown files.
    """
    if tracker:
        tracker.log_method("markdown_pandoc")

    try:
        # For markdown, we can read directly and optionally use pandoc
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try pandoc for better formatting
        try:
            cmd = ['pandoc', file_path, '-t', 'plain']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            if tracker:
                tracker.log_method("markdown_raw")
            pass
        
        # Fallback to raw content
        logger.info(f"Extracted {len(content)} characters from markdown")
        return content
        
    except Exception as e:
        logger.error(f"Markdown extraction failed: {e}")
        raise

async def extract_text_from_image(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from image files using OCR.
    """
    if tracker:
        tracker.log_method("image_ocr")

    try:
        img = Image.open(file_path)
        return await extract_text_from_image_data(img)
        
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        raise

async def extract_text_from_image_data(img: Image.Image) -> str:
    """
    Extract text from PIL Image using PaddleOCR.
    """
    
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Initialize OCR
        ocr = get_ocr_instance()
        
        # Run OCR
        result = ocr.ocr(img_array, cls=True)
        
        # Extract text from results
        text_parts = []
        if result and result[0]:  # Check if results exist
            for line in result[0]:
                if line and len(line) > 1 and line[1][0]:  # Check if text exists
                    text_parts.append(line[1][0])
        
        extracted_text = '\n'.join(text_parts)
        logger.info(f"OCR extracted {len(extracted_text)} characters from image")
        return extracted_text
        
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        raise

async def extract_text_by_pages_from_pdf(file_path: str, tracker: Optional[ResourceTracker] = None) -> List[str]:
    """
    Extract text from PDF page by page, returning a list of page texts.
    
    Args:
        file_path: Path to the PDF file
        tracker: Optional resource tracker instance
    
    Returns:
        List of strings, where each string contains the text from one page
    """
    
    try:
        # Add timeout for PDF processing (60 seconds for large textbooks)
        pages_text = await asyncio.wait_for(
            extract_text_by_pages_pdfminer(file_path, tracker), 
            timeout=60.0
        )
        
        if pages_text and any(len(page.strip()) > 10 for page in pages_text):  # Reasonable amount of text found
            logger.info(f"Successfully extracted text from {len(pages_text)} pages using pdfminer")
            if tracker:
                tracker.log_method("pdfminer_pages")
            return pages_text
        
        # Check if OCR is enabled before attempting OCR
        if not settings.ENABLE_IMAGE_OCR:
            logger.warning("PDF appears to be scanned but OCR is disabled. Returning minimal text extracted.")
            if tracker:
                tracker.log_method("pdfminer_no_ocr_pages")
            return pages_text if pages_text else [""]
        
        logger.info("PDF appears to be scanned or has minimal text, trying OCR")
        
        # If minimal text, try OCR approach
        return await asyncio.wait_for(
            extract_text_by_pages_pdf_ocr(file_path, tracker),
            timeout=180.0  # OCR takes much longer for textbooks
        )
        
    except Exception as e:
        logger.warning(f"pdfminer page extraction failed: {e}. Trying OCR approach")
        
        # Check if OCR is enabled before attempting OCR fallback
        if not settings.ENABLE_IMAGE_OCR:
            logger.error("PDF processing failed and OCR is disabled. Cannot extract text.")
            raise Exception("PDF processing failed and OCR is disabled. Enable OCR with ENABLE_IMAGE_OCR=true to handle scanned PDFs.")
        
        return await asyncio.wait_for(
            extract_text_by_pages_pdf_ocr(file_path, tracker),
            timeout=180.0
        )

async def extract_text_by_pages_pdfminer(file_path: str, tracker: Optional[ResourceTracker] = None) -> List[str]:
    """
    Extract text from PDF page by page using pdfminer.six.
    """
    pages_text = []
    
    try:
        with open(file_path, 'rb') as file:
            resource_manager = PDFResourceManager()
            laparams = LAParams()
            
            for page_num, page in enumerate(PDFPage.get_pages(file, check_extractable=True)):
                # Create a text converter for this page
                output_string = StringIO()
                converter = TextConverter(resource_manager, output_string, laparams=laparams)
                page_interpreter = PDFPageInterpreter(resource_manager, converter)
                
                # Process the page
                page_interpreter.process_page(page)
                text = output_string.getvalue()
                
                # Clean up
                converter.close()
                output_string.close()
                
                pages_text.append(text.strip() if text.strip() else "")
                
        logger.info(f"pdfminer extracted text from {len(pages_text)} pages")
        return pages_text
        
    except Exception as e:
        logger.error(f"pdfminer page extraction failed: {e}")
        raise

async def extract_text_by_pages_pdf_ocr(file_path: str, tracker: Optional[ResourceTracker] = None) -> List[str]:
    """
    Extract text from PDF page by page using OCR (for scanned PDFs).
    """
    if tracker:
        tracker.log_method("pdf_ocr_pymupdf_pages")

    if PYMUPDF_AVAILABLE:
        try:
            # Convert PDF to images using PyMuPDF
            doc = fitz.open(file_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Run OCR on the image
                page_text = await extract_text_from_image_data(img)
                pages_text.append(page_text if page_text else "")
            
            doc.close()
            
            logger.info(f"OCR extracted text from {len(pages_text)} pages")
            return pages_text
            
        except Exception as e:
            logger.error(f"PyMuPDF OCR page extraction failed: {e}")
            # Fallback to pdftoppm
            return await extract_text_by_pages_pdf_pdftoppm(file_path, tracker)
    else:
        # Fallback to pdftoppm if PyMuPDF is not available
        return await extract_text_by_pages_pdf_pdftoppm(file_path, tracker)

async def extract_text_by_pages_pdf_pdftoppm(file_path: str, tracker: Optional[ResourceTracker] = None) -> List[str]:
    """
    Extract text from PDF page by page using pdftoppm + OCR as fallback.
    """
    if tracker:
        tracker.log_method("pdf_ocr_pdftoppm_pages")
    
    try:
        # Convert PDF to images using pdftoppm
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                'pdftoppm', 
                '-png', 
                file_path, 
                os.path.join(temp_dir, 'page')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"pdftoppm failed: {result.stderr}")
            
            # Process each generated image
            pages_text = []
            image_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
            
            for image_file in image_files:
                image_path = os.path.join(temp_dir, image_file)
                page_text = await extract_text_from_image(image_path)
                pages_text.append(page_text if page_text else "")
            
            logger.info(f"pdftoppm + OCR extracted text from {len(pages_text)} pages")
            return pages_text
            
    except Exception as e:
        logger.error(f"pdftoppm + OCR page extraction failed: {e}")
        raise

async def extract_text_from_epub(file_path: str, tracker: Optional[ResourceTracker] = None) -> str:
    """
    Extract text from EPUB file.
    
    Args:
        file_path: Path to the EPUB file
        tracker: Optional resource tracker instance
    
    Returns:
        Extracted text as string
    """
    if not EPUB_AVAILABLE:
        raise Exception("EPUB processing is not available. Install ebooklib and beautifulsoup4 to enable EPUB support.")
    
    if tracker:
        tracker.log_method("epub_extraction")
    
    try:
        # Read the EPUB file
        book = epub.read_epub(file_path)
        text_content = []
        
        # Get all document items (chapters, sections, etc.)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Extract text from various HTML elements
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if text:
                text_content.append(text)
        
        # Join all text content
        full_text = '\n\n'.join(text_content)
        
        logger.info(f"EPUB extracted {len(full_text)} characters from {len(text_content)} sections")
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"EPUB extraction failed: {e}")
        raise

async def extract_text_by_chapters_from_epub(file_path: str, tracker: Optional[ResourceTracker] = None) -> List[str]:
    """
    Extract text from EPUB file chapter by chapter.
    
    Args:
        file_path: Path to the EPUB file
        tracker: Optional resource tracker instance
    
    Returns:
        List of strings, where each string contains the text from one chapter/section
    """
    if not EPUB_AVAILABLE:
        raise Exception("EPUB processing is not available. Install ebooklib and beautifulsoup4 to enable EPUB support.")
    
    if tracker:
        tracker.log_method("epub_chapter_extraction")
    
    try:
        # Read the EPUB file
        book = epub.read_epub(file_path)
        chapters_text = []
        
        # Get all document items (chapters, sections, etc.)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Extract text from various HTML elements
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            chapters_text.append(text if text else "")
        
        logger.info(f"EPUB extracted text from {len(chapters_text)} chapters")
        return chapters_text
        
    except Exception as e:
        logger.error(f"EPUB chapter extraction failed: {e}")
        raise
