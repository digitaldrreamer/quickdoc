from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys

from .routes import router

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Processing Service",
    description="A microservice for extracting text from documents and converting HTML/Markdown to PDF",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "docs-processing"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Document Processing Service",
        "endpoints": {
            "extract": "POST /extract - Upload a document to extract text",
            "convert_to_pdf": "POST /convert-to-pdf - Convert HTML or Markdown content to PDF with enhanced options",
            "convert_to_pdf_upload": "POST /convert-to-pdf-upload - Upload and convert HTML/Markdown file to PDF",
            "pdf_settings": "GET/POST /pdf-settings - Get or set PDF generation settings",
            "supported_formats": "GET /supported-formats - Get supported file formats",
            "health": "GET /health - Health check"
        },
        "features": {
            "markdown_processors": ["Mistune (enhanced with plugins)", "Pandoc (for complex content)"],
            "pdf_features": [
                "Automatic complexity detection",
                "Configurable page settings",
                "Custom CSS support",
                "PDF metadata configuration",
                "Table of contents generation",
                "Multiple priority modes (speed/quality/auto)"
            ],
            "supported_formats": [
                "HTML to PDF",
                "Markdown to PDF (with enhanced plugin support)",
                "Document text extraction (PDF, DOCX, ODT, RTF, Images)"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
