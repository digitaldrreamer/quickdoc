from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
import logging
import tempfile
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import time

from .processor import extract_text_from_doc, extract_text_by_pages_from_pdf
from .utils import get_file_type, is_supported_file_type
from .metrics import ResourceTracker
from .pdf_converter import PDFConverter, Priority, ContentType

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported file types
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.odt', '.rtf', '.md', '.markdown',
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
}

# Pydantic models for PDF conversion request
class PDFConversionRequest(BaseModel):
    content: str = Field(..., description="HTML or Markdown content to convert")
    content_type: ContentType = Field(..., description="Type of content: html or markdown")
    priority: Priority = Field(default=Priority.AUTO, description="Processing priority: auto, speed, or quality")
    css_content: Optional[str] = Field(default=None, description="Optional CSS styling")
    weasyprint_options: Optional[Dict[str, Any]] = Field(default=None, description="Optional weasyprint configuration")
    
    # Enhanced WeasyPrint configuration options
    page_size: Optional[str] = Field(default="A4", description="Page size (A4, Letter, etc.)")
    page_orientation: Optional[str] = Field(default="portrait", description="Page orientation (portrait, landscape)")
    page_margins: Optional[Dict[str, str]] = Field(default=None, description="Page margins (top, right, bottom, left)")
    base_url: Optional[str] = Field(default=None, description="Base URL for resolving relative URLs")
    
    # Font configuration
    font_family: Optional[str] = Field(default=None, description="Default font family")
    font_size: Optional[str] = Field(default="12pt", description="Default font size")
    
    # Advanced options
    optimize_images: Optional[bool] = Field(default=True, description="Optimize images for smaller file size")
    pdf_version: Optional[str] = Field(default="1.7", description="PDF version")
    pdf_identifier: Optional[bool] = Field(default=True, description="Include PDF identifier")
    pdf_title: Optional[str] = Field(default=None, description="PDF document title")
    pdf_author: Optional[str] = Field(default=None, description="PDF document author")
    pdf_subject: Optional[str] = Field(default=None, description="PDF document subject")
    pdf_keywords: Optional[str] = Field(default=None, description="PDF document keywords")
    
    # Table of contents
    generate_toc: Optional[bool] = Field(default=False, description="Generate table of contents")
    toc_depth: Optional[int] = Field(default=3, description="TOC depth (1-6)")

class PDFConversionResponse(BaseModel):
    success: bool = Field(..., description="Whether conversion was successful")
    message: str = Field(..., description="Status message")
    conversion_stats: Dict[str, Any] = Field(..., description="Conversion statistics")
    file_size: int = Field(..., description="Size of generated PDF in bytes")
    processing_time: float = Field(..., description="Processing time in seconds")
    
class PDFSettingsRequest(BaseModel):
    """Request model for PDF settings configuration"""
    default_page_size: Optional[str] = Field(default="A4", description="Default page size")
    default_font_family: Optional[str] = Field(default="DejaVu Sans", description="Default font family")
    default_font_size: Optional[str] = Field(default="12pt", description="Default font size")
    default_margins: Optional[Dict[str, str]] = Field(default=None, description="Default page margins")
    enable_toc: Optional[bool] = Field(default=False, description="Enable table of contents by default")
    image_optimization: Optional[bool] = Field(default=True, description="Enable image optimization")
    pdf_version: Optional[str] = Field(default="1.7", description="PDF version")

class PDFPagesResponse(BaseModel):
    """Response model for page-by-page PDF text extraction"""
    pages: List[str] = Field(..., description="List of text content for each page")
    page_count: int = Field(..., description="Total number of pages")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension")
    metrics: Dict[str, Any] = Field(..., description="Processing metrics")

@router.post("/extract")
async def extract_text_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract text from uploaded document.
    
    Accepts: PDF, DOCX, ODT, RTF, Markdown, or image files
    Returns: JSON with extracted text and processing metrics
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check if file type is supported
    if not is_supported_file_type(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Create temporary file
    temp_file_path = ""
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Create temporary file
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Processing file: {file.filename} (type: {suffix}, size: {file_size} bytes)")
        
        # Track metrics during text extraction
        with ResourceTracker() as tracker:
            extracted_text = await extract_text_from_doc(temp_file_path, file.filename, tracker)
        
        metrics = tracker.get_metrics(file_size_bytes=file_size)
        logger.info(f"Extraction metrics for {file.filename}: {metrics}")

        if not extracted_text or not extracted_text.strip():
            logger.warning(f"No text extracted from file: {file.filename}")
            return {
                "text": "",
                "warning": "No text could be extracted from the document",
                "metrics": metrics
            }
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {file.filename}")
        
        return {
            "text": extracted_text,
            "filename": file.filename,
            "file_type": suffix,
            "character_count": len(extracted_text),
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")

@router.post("/extract-pages", response_model=PDFPagesResponse)
async def extract_text_by_pages_endpoint(file: UploadFile = File(...)) -> PDFPagesResponse:
    """
    Extract text from uploaded PDF page by page.
    
    Accepts: PDF files only
    Returns: JSON with array of page texts and processing metrics
    
    Processing Limits:
    - File size: Maximum 50MB
    - Text-based PDFs: 60 second timeout
    - Scanned PDFs (OCR): 180 second timeout
    
    Recommendations for Large Documents:
    - For textbooks with 500+ pages, consider processing in batches
    - Very large scanned documents (1000+ pages) may require background processing
    - Complex academic PDFs with equations/diagrams may take longer to process
    
    Response Format:
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
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check if file is PDF
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != '.pdf':
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported for page-by-page extraction"
        )
    
    # Check file size limit (use existing limit from settings)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size for page-by-page extraction is 50MB."
        )
    
    # Create temporary file
    temp_file_path = ""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Processing PDF for page-by-page extraction: {file.filename} (size: {file_size} bytes)")
        
        # Track metrics during text extraction
        with ResourceTracker() as tracker:
            pages_text = await extract_text_by_pages_from_pdf(temp_file_path, tracker)
        
        metrics = tracker.get_metrics(file_size_bytes=file_size)
        logger.info(f"Page-by-page extraction metrics for {file.filename}: {metrics}")

        if not pages_text or not any(page.strip() for page in pages_text):
            logger.warning(f"No text extracted from any page of file: {file.filename}")
            return PDFPagesResponse(
                pages=[""],
                page_count=1,
                filename=file.filename,
                file_type=file_ext,
                metrics=metrics
            )
        
        logger.info(f"Successfully extracted text from {len(pages_text)} pages of {file.filename}")
        
        return PDFPagesResponse(
            pages=pages_text,
            page_count=len(pages_text),
            filename=file.filename,
            file_type=file_ext,
            metrics=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing PDF file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_extensions": sorted(list(SUPPORTED_EXTENSIONS)),
        "description": "Upload documents in any of these formats for text extraction"
    }

@router.post("/convert-to-pdf")
async def convert_to_pdf_endpoint(request: PDFConversionRequest) -> Response:
    """
    Convert HTML or Markdown content to PDF with enhanced configuration options.
    
    Accepts: HTML or Markdown content with extensive configuration options
    Returns: PDF file as binary response
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    with ResourceTracker() as tracker:
        try:
            # Create enhanced CSS based on request parameters
            enhanced_css = _build_enhanced_css(request)
            
            # Build WeasyPrint options
            weasyprint_options = _build_weasyprint_options(request)
            
            # Initialize PDF converter
            converter = PDFConverter()
            
            # Convert content to PDF
            pdf_bytes, stats = await converter.convert_to_pdf(
                content=request.content,
                content_type=request.content_type,
                priority=request.priority,
                css_content=enhanced_css,
                weasyprint_options=weasyprint_options,
                tracker=tracker
            )
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"document_{timestamp}.pdf"
            
            # Return PDF response
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "X-Conversion-Stats": json.dumps(stats)
                }
            )
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

@router.post("/pdf-settings")
async def set_pdf_settings(settings: PDFSettingsRequest) -> Dict[str, Any]:
    """
    Configure default PDF generation settings.
    
    Accepts: PDF settings configuration
    Returns: Confirmation of settings applied
    """
    try:
        # In a real implementation, you'd save these to a database or config file
        # For now, we'll just return the settings that would be applied
        return {
            "message": "PDF settings configured successfully",
            "settings": settings.dict(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to configure PDF settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure PDF settings: {str(e)}")

@router.get("/pdf-settings")
async def get_pdf_settings() -> Dict[str, Any]:
    """
    Get current PDF generation settings.
    
    Returns: Current PDF settings configuration
    """
    try:
        # Return default settings (in a real implementation, load from config)
        default_settings = {
            "default_page_size": "A4",
            "default_font_family": "DejaVu Sans", 
            "default_font_size": "12pt",
            "default_margins": None,
            "enable_toc": False,
            "image_optimization": True,
            "pdf_version": "1.7"
        }
        return {
            "settings": default_settings,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to get PDF settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get PDF settings: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.post("/convert-to-pdf-upload")
async def convert_to_pdf_upload_endpoint(
    file: UploadFile = File(...),
    content_type: ContentType = Form(...),
    priority: Priority = Form(default=Priority.AUTO),
    css_content: Optional[str] = Form(default=None)
) -> Response:
    """
    Convert uploaded HTML or Markdown file to PDF.
    
    This endpoint accepts file uploads and converts them to PDF using the same
    intelligent routing logic as the content-based endpoint.
    
    Args:
        file: Uploaded HTML or Markdown file
        content_type: Type of content (html or markdown)
        priority: Processing priority
        css_content: Optional CSS styling
        
    Returns:
        PDF file as binary response with conversion statistics in headers
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if content_type == ContentType.MARKDOWN and file_ext not in ['.md', '.markdown']:
        raise HTTPException(
            status_code=400, 
            detail="File extension should be .md or .markdown for markdown content"
        )
    elif content_type == ContentType.HTML and file_ext not in ['.html', '.htm']:
        raise HTTPException(
            status_code=400, 
            detail="File extension should be .html or .htm for HTML content"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Decode content
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400, 
                detail="File must be UTF-8 encoded"
            )
        
        # Validate content length (max 10MB)
        max_content_length = 10 * 1024 * 1024  # 10MB
        if len(content) > max_content_length:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {max_content_length} bytes"
            )
        
        converter = PDFConverter()
        
        # Track metrics during conversion
        with ResourceTracker() as tracker:
            pdf_bytes, conversion_stats = await converter.convert_to_pdf(
                content=content_str,
                content_type=content_type,
                priority=priority,
                css_content=css_content,
                tracker=tracker
            )
        
        # Get resource metrics
        resource_metrics = tracker.get_metrics(len(content))
        
        # Combine stats
        full_stats = {
            **conversion_stats,
            "resource_metrics": resource_metrics,
            "original_filename": file.filename
        }
        
        logger.info(f"PDF conversion from upload successful: {full_stats}")
        
        # Create response with PDF content
        base_filename = os.path.splitext(file.filename)[0]
        response = Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={base_filename}.pdf",
                "X-Conversion-Stats": str(full_stats),
                "X-Converter-Used": conversion_stats.get("converter_used", "unknown"),
                "X-Processing-Time": str(conversion_stats.get("conversion_time_seconds", 0)),
                "X-Content-Type": content_type.value,
                "X-Priority": priority.value,
                "X-Complexity-Detected": str(conversion_stats.get("complexity_detected", False)),
                "X-PDF-Size": str(len(pdf_bytes)),
                "X-Original-Filename": file.filename
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF conversion from upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF conversion failed: {str(e)}"
        )

def _build_enhanced_css(request: PDFConversionRequest) -> str:
    """Build enhanced CSS based on request parameters"""
    
    # Base CSS
    base_css = f"""
    @page {{
        size: {request.page_size or 'A4'};
        margin: {_format_margins(request.page_margins)};
        {f"orientation: {request.page_orientation};" if request.page_orientation != "portrait" else ""}
    }}
    
    body {{
        font-family: {request.font_family or "'DejaVu Sans', Arial, sans-serif"};
        font-size: {request.font_size or '12pt'};
        line-height: 1.4;
        color: #333;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: #2c3e50;
        page-break-after: avoid;
    }}
    
    table {{
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 1em;
    }}
    
    th, td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }}
    
    th {{
        background-color: #f2f2f2;
    }}
    
    code {{
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'DejaVu Sans Mono', monospace;
    }}
    
    pre {{
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }}
    
    blockquote {{
        border-left: 4px solid #ccc;
        margin-left: 0;
        padding-left: 1em;
        color: #666;
    }}
    """
    
    # Add table of contents styles if requested
    if request.generate_toc:
        toc_css = """
        .toc {
            page-break-after: always;
            border-bottom: 1px solid #ccc;
            padding-bottom: 1em;
            margin-bottom: 1em;
        }
        
        .toc h2 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        
        .toc li {
            margin-bottom: 0.5em;
        }
        
        .toc a {
            text-decoration: none;
            color: #333;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        """
        base_css += toc_css
    
    # Combine with user CSS
    if request.css_content:
        base_css += f"\n/* User CSS */\n{request.css_content}"
    
    return base_css

def _format_margins(margins: Optional[Dict[str, str]]) -> str:
    """Format page margins for CSS"""
    if not margins:
        return "1in"
    
    top = margins.get("top", "1in")
    right = margins.get("right", "1in")
    bottom = margins.get("bottom", "1in")
    left = margins.get("left", "1in")
    
    return f"{top} {right} {bottom} {left}"

def _build_weasyprint_options(request: PDFConversionRequest) -> Dict[str, Any]:
    """Build WeasyPrint options based on request parameters"""
    
    options = request.weasyprint_options or {}
    
    # Add PDF metadata
    if request.pdf_title:
        options["pdf_title"] = request.pdf_title
    if request.pdf_author:
        options["pdf_author"] = request.pdf_author
    if request.pdf_subject:
        options["pdf_subject"] = request.pdf_subject
    if request.pdf_keywords:
        options["pdf_keywords"] = request.pdf_keywords
    
    # Add PDF version
    if request.pdf_version:
        options["pdf_version"] = request.pdf_version
    
    # Add PDF identifier
    if request.pdf_identifier is not None:
        options["pdf_identifier"] = request.pdf_identifier
    
    # Add image optimization
    if request.optimize_images is not None:
        options["optimize_images"] = request.optimize_images
    
    return options
