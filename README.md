# QuickDoc - Open Source Document & AI Processor

An intelligent, configurable microservice for document text extraction, AI-powered summarization, text embedding, and token counting. Built with FastAPI and designed for scalability, QuickDoc lets you enable only the features you need to optimize resource usage.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, ODT, RTF, Markdown, EPUB, and images (JPG, PNG, BMP, TIFF)
- **Page-by-Page Extraction**: Extract text from PDFs page by page or as complete document
- **Chapter-by-Chapter Extraction**: Extract text from EPUBs chapter by chapter or as complete book
- **Intelligent OCR**: Automatic text extraction from scanned PDFs and images using PaddleOCR
- **Configurable Processing**: Enable/disable specific document types to save resources

### AI Util Services
- **Text Summarization**: Advanced summarization with configurable quality levels using transformer models
- **Text Embeddings**: Generate semantic embeddings for texts and documents
- **Token Counting**: Accurate token counting for Llama 3, Mistral, and Gemini models
- **Async Processing**: Non-blocking AI operations with queue management

### Configuration & Resource Management
- **Modular Features**: Enable only the services you need
- **Resource Optimization**: Conditional model loading based on configuration
- **Configurable Models**: Choose your preferred AI models via environment variables
- **Production Ready**: Docker support with health checks and proper logging

---

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/digitaldrreamer/quickdoc.git
   cd quickdoc
   ```

2. **Configure your deployment**
   ```bash
   cp env.example .env
   # Edit .env to enable/disable features as needed
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Test the service**
   ```bash
   # Test document extraction
   echo "Hello, QuickDoc!" > test.md
   curl -F "file=@test.md" http://localhost:8002/extract
   
   # Check service status
   curl http://localhost:8002/health
   ```

The service will be available at `http://localhost:8002` with interactive documentation at `http://localhost:8002/docs`. You can set the port in `.env`

### Option 2: Manual Installation

#### Prerequisites

**System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    pandoc poppler-utils libmagic1 tesseract-ocr \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libgomp1

# macOS
brew install pandoc poppler libmagic tesseract
```

**Python Setup:**
```bash
# Clone and setup
git clone https://github.com/digitaldrreamer/quickdoc.git
cd quickdoc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Configuration

```bash
# Copy and configure environment
cp env.example .env
# Edit .env file with your preferences
```

#### Run the Service

```bash
# Development
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Production
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8002
```

---

## Configuration

QuickDoc is highly configurable through environment variables. Copy `env.example` to `.env` and customize:

### Core Features
```bash
# Enable/disable major components
ENABLE_SUMMARIZATION=true          # AI text summarization
ENABLE_EMBEDDING_MODEL=true        # Text embedding generation  
ENABLE_TOKEN_COUNTING=true         # Token counting for various models
ENABLE_DOCUMENT_PROCESSING=true    # Document text extraction
```

### Document Processing
```bash
# Fine-grained document type control
ENABLE_PDF_PROCESSING=true         # PDF text extraction & OCR
ENABLE_DOCX_PROCESSING=true        # Word/ODT/RTF processing
ENABLE_IMAGE_OCR=true              # Image text extraction
ENABLE_MARKDOWN_PROCESSING=true    # Markdown processing
```

### AI Models
```bash
# Specify which models to use
SUMMARIZATION_MODEL=google/flan-t5-small    # Hugging Face model for summarization
EMBEDDING_MODEL=all-MiniLM-L6-v2           # Sentence transformer model
```

### Resource Optimization Examples

**Minimal Deployment (Text extraction only):**
```bash
ENABLE_SUMMARIZATION=false
ENABLE_EMBEDDING_MODEL=false
ENABLE_TOKEN_COUNTING=false
ENABLE_IMAGE_OCR=false
```

**PDF-only Service:**
```bash
ENABLE_DOCX_PROCESSING=false
ENABLE_IMAGE_OCR=false
ENABLE_MARKDOWN_PROCESSING=false
```

**AI-only Service (no document processing):**
```bash
ENABLE_DOCUMENT_PROCESSING=false
```

---

## API Documentation

### Document Conversion Endpoints

#### `POST /extract`
Extract text from documents.

```bash
curl -X POST -F "file=@document.pdf" http://localhost:8002/extract
```

**Response:**
```json
{
  "text": "Extracted text content...",
  "filename": "document.pdf",
  "file_type": ".pdf",
  "character_count": 1250,
  "metrics": {
    "processing_duration_ms": 150.2,
    "memory_usage_mb": 75.3,
    "processing_method": "pdfminer"
  }
}
```

#### `POST /convert-to-pdf`
Convert documents to PDF format.

### AI Service Endpoints

#### `POST /ai/embed/text`
Generate embeddings for text.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "normalize": true}' \
  http://localhost:8002/ai/embed/text
```

#### `POST /ai/embed/document`
Extract text from document and generate embeddings.

```bash
curl -X POST -F "file=@document.pdf" http://localhost:8002/ai/embed/document
```

#### `POST /ai/summarize`
Summarize text with configurable quality.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Long text to summarize...", "max_length": 150, "quality": "high"}' \
  http://localhost:8002/ai/summarize
```

#### `POST /ai/tokens/count/{model}`
Count tokens for specific models (llama3, mistral, gemini).

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Text to count tokens for"}' \
  http://localhost:8002/ai/tokens/count/llama3
```

### Utility Endpoints

- `GET /health` - Service health check with feature status
- `GET /` - API overview and available endpoints
- `GET /docs` - Interactive API documentation (Swagger UI)

---

## Deployment

### Docker Deployment

**Basic deployment:**
```bash
docker-compose up -d
```

**With custom configuration:**
```bash
# Create custom .env file
cp env.example .env
# Edit .env with your settings
docker-compose up -d
```

### Production Deployment

**Using Docker with resource limits:**
```yaml
version: '3.8'
services:
  quickdoc:
    image: quickdoc:latest
    ports:
      - "8002:8002"
    env_file: .env
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Environment Variables for Production:**
```bash
# Resource optimization
MAX_FILE_SIZE_MB=50
SUMMARIZATION_TIMEOUT=600
MAX_SUMMARIZATION_QUEUE_SIZE=50
LOG_LEVEL=WARNING

# Security (if using external APIs)
HUGGING_FACE_HUB_TOKEN=your_secure_token
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quickdoc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quickdoc
  template:
    metadata:
      labels:
        app: quickdoc
    spec:
      containers:
      - name: quickdoc
        image: quickdoc:latest
        ports:
        - containerPort: 8002
        env:
        - name: ENABLE_SUMMARIZATION
          value: "true"
        - name: ENABLE_EMBEDDING_MODEL
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: quickdoc-service
spec:
  selector:
    app: quickdoc
  ports:
  - port: 80
    targetPort: 8002
  type: LoadBalancer
```

---

## Development

### Setting up Development Environment

```bash
# Clone and setup
git clone https://github.com/digitaldrreamer/quickdoc.git
cd quickdoc

# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run in development mode
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

### Running Tests

```bash
# Run the test suite
python -m pytest

# Test specific endpoints
python test_pdf_endpoints.py
python test_enhanced_pdf.py
```

### Code Quality

The project follows Google Python Style Guide and includes:
- Type hints throughout the codebase
- Comprehensive error handling
- Structured logging
- Resource tracking and metrics
- Async/await patterns for scalability

---

## Technology Stack

- **Framework**: FastAPI 0.104+
- **AI/ML**: 
  - Transformers 4.41+ (summarization)
  - Sentence Transformers 2.7+ (embeddings)
  - PaddleOCR 2.7+ (OCR)
- **Document Processing**: 
  - PDFMiner.six (PDF text extraction)
  - Pandoc (document conversion)
  - PyMuPDF (PDF rendering)
- **Infrastructure**: 
  - Docker & Docker Compose
  - Uvicorn/Gunicorn (ASGI servers)
  - Pydantic (configuration & validation)

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Follow the coding standards**: Google Python Style Guide
5. **Submit a pull request**

### Development Guidelines

- Add type hints to all functions
- Include docstrings for public methods
- Write tests for new features
- Update documentation for API changes
- Use Better Comments style for inline comments
- Use good branch names so one can understand at first glance

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PaddleOCR** for excellent OCR capabilities
- **Hugging Face Transformers** for state-of-the-art NLP models
- **FastAPI** for being so excellent
- **The open source community** for inspiration and tools

---

## Support

- **Documentation**: Check `/docs` endpoint when service is running
- **Issues**: Please report bugs via GitHub Issues
- **Discussions**: Start discussions for feature requests and questions