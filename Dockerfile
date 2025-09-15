# Stage 1: Builder - Installs all dependencies and builds the environment
FROM python:3.10-slim AS builder

# Set environment variables to prevent buffering and disable pip cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# Install all necessary system dependencies (both build-time and run-time)
# This layer is cached by Docker if the list doesn't change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build-time dependencies (for compiling Python packages)
    build-essential \
    python3-dev \
    libtesseract-dev \
    libxrender-dev \
    # Run-time dependencies (needed by the application to function)
    pandoc \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    # EPUB processing dependencies
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    # Utilities
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies.
# This layer is cached if requirements.txt doesn't change, speeding up rebuilds.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# -----------------------------------------------------------------------------


# Stage 2: Final Image - A lean, production-ready image
FROM python:3.10-slim AS final

WORKDIR /app

# Install ONLY the necessary run-time system dependencies.
# This makes the final image significantly smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    # EPUB processing runtime dependencies
    libxml2 \
    libxslt1.1 \
    zlib1g \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./app ./app

# Note: This application now supports:
# - PDF text extraction (page-by-page and full text)
# - EPUB text extraction (chapter-by-chapter and full text)
# - DOCX, ODT, RTF, Markdown, and image processing
# - AI-powered summarization and embedding services

# Create a non-root user and give it ownership of the app directory.
# This is a critical security best practice.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Add a HEALTHCHECK to let Docker know if the app is running correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# Expose port and define the command to run the application
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]