from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from contextlib import asynccontextmanager
import asyncio
from typing import Optional

from .routes import router as doc_conversion_router
from .api.ai_routes import router as ai_router
from .api.embedding_routes import router as embedding_router
from .core.config import settings

# * Conditional imports based on configuration
if settings.ENABLE_EMBEDDING_MODEL:
    from .services.embedding_service import EmbeddingService

if settings.ENABLE_SUMMARIZATION:
    from .services.summarization_service import OptimizedSummarizer

if settings.ENABLE_TOKEN_COUNTING:
    from .services.token_service import TokenCounter

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()), 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                   stream=sys.stderr)
logger = logging.getLogger(__name__)


# Lifespan manager to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading enabled services...")
    loop = asyncio.get_event_loop()
    
    # * Initialize services based on configuration
    futures = []
    
    if settings.ENABLE_EMBEDDING_MODEL:
        logger.info("Loading embedding service...")
        futures.append(("embedder", loop.run_in_executor(None, lambda: EmbeddingService(settings.EMBEDDING_MODEL))))
    else:
        logger.info("Embedding service disabled - skipping model loading")
        
    if settings.ENABLE_SUMMARIZATION:
        logger.info("Loading summarization service...")
        futures.append(("summarizer", loop.run_in_executor(None, lambda: OptimizedSummarizer(settings.SUMMARIZATION_MODEL))))
    else:
        logger.info("Summarization service disabled - skipping model loading")
        
    if settings.ENABLE_TOKEN_COUNTING:
        logger.info("Loading token counting service...")
        futures.append(("tokenizer", loop.run_in_executor(None, TokenCounter)))
    else:
        logger.info("Token counting service disabled - skipping model loading")

    # * Wait for enabled services to load
    if futures:
        results = await asyncio.gather(*[future for _, future in futures])
        for (service_name, _), result in zip(futures, results):
            setattr(app.state, service_name, result)
    
    # * Initialize optional services with None if disabled
    if not settings.ENABLE_EMBEDDING_MODEL:
        app.state.embedder = None
    if not settings.ENABLE_SUMMARIZATION:
        app.state.summarizer = None
    if not settings.ENABLE_TOKEN_COUNTING:
        app.state.tokenizer = None

    logger.info("All enabled services loaded.")

    # * Start summarization worker only if summarization is enabled
    if settings.ENABLE_SUMMARIZATION and hasattr(app.state, 'summarizer'):
        app.state.summarization_queue = asyncio.Queue(maxsize=settings.MAX_SUMMARIZATION_QUEUE_SIZE)
        app.state.summarization_worker = asyncio.create_task(
            summarization_worker(app.state.summarization_queue, app.state.summarizer)
        )
        logger.info("Summarization worker started.")
    else:
        app.state.summarization_queue = None
        app.state.summarization_worker = None

    yield

    # Cleanup
    if hasattr(app.state, 'summarization_worker') and app.state.summarization_worker:
        app.state.summarization_worker.cancel()
        logger.info("Summarization worker stopped.")
    logger.info("Application shutdown.")


# Background worker for summarization queue
async def summarization_worker(queue: asyncio.Queue, summarizer):
    while True:
        try:
            request_data, future = await queue.get()
            try:
                summary = await summarizer.summarize(**request_data)
                future.set_result(summary)
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                future.set_exception(e)
            finally:
                queue.task_done()
        except asyncio.CancelledError:
            break


# Create FastAPI app
app = FastAPI(
    title="QuickDoc - Open Source Document & AI Processor",
    description="A microservice for text extraction, PDF conversion, embedding, summarization, and tokenization with configurable features.",
    version="2.1.0",
    lifespan=lifespan
)


# Dynamic CORS class (handles subdomain-based origin allow)
class DynamicCORS(CORSMiddleware):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._known_origins = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            origin_header = dict(scope["headers"]).get(b"origin")
            if origin_header:
                origin = origin_header.decode("latin1")
                # * Allow localhost and common development patterns
                if any(allowed in origin for allowed in ["localhost", "127.0.0.1", "0.0.0.0"]) or origin.endswith(".kuroconnect.com"):
                    if origin not in self._known_origins:
                        self.allow_origins.append(origin)
                        self._known_origins.add(origin)
                        logger.info(f"Dynamically allowed CORS origin: {origin}")
        return await super().__call__(scope, receive, send)


# Register CORS middleware
app.add_middleware(
    DynamicCORS,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Common development ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers conditionally
if settings.ENABLE_DOCUMENT_PROCESSING:
    app.include_router(doc_conversion_router, tags=["Document Conversion"])
    logger.info("Document conversion routes enabled")
else:
    logger.info("Document processing disabled - skipping document conversion routes")

# * Always include AI routes, but they'll check for service availability internally
app.include_router(ai_router, prefix="/ai", tags=["AI Services"])

# * Include embedding routes if embedding model is enabled
if settings.ENABLE_EMBEDDING_MODEL:
    app.include_router(embedding_router, tags=["Document Embeddings"])
    logger.info("Document embedding routes enabled")
else:
    logger.info("Embedding model disabled - skipping embedding routes")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )


# Health check
@app.get("/health", tags=["Utilities"])
async def health_check():
    """Health check endpoint with service status information."""
    status = {
        "status": "healthy",
        "service": "quickdoc-processor",
        "version": "2.1.0",
        "features": {
            "document_processing": settings.ENABLE_DOCUMENT_PROCESSING,
            "summarization": settings.ENABLE_SUMMARIZATION,
            "embedding": settings.ENABLE_EMBEDDING_MODEL,
            "token_counting": settings.ENABLE_TOKEN_COUNTING,
        }
    }
    return status


# Root endpoint with dynamic documentation
@app.get("/", tags=["Utilities"])
async def root():
    endpoints = {
        "message": "QuickDoc - Open Source Document & AI Processor",
        "documentation": "/docs",
        "health": "/health",
    }
    
    if settings.ENABLE_DOCUMENT_PROCESSING:
        endpoints["document_conversion_endpoints"] = {
            "extract_text": "POST /extract",
            "convert_to_pdf": "POST /convert-to-pdf",
        }
    
    ai_endpoints = {}
    if settings.ENABLE_EMBEDDING_MODEL:
        ai_endpoints.update({
            "embed_text": "POST /ai/embed/text",
            "embed_document": "POST /ai/embed/document",
        })
        endpoints["document_embedding_endpoints"] = {
            "embed_document": "POST /embed/document",
            "embed_text_chunks": "POST /embed/text",
            "get_strategies": "GET /embed/strategies",
            "health_check": "GET /embed/health",
        }
    if settings.ENABLE_SUMMARIZATION:
        ai_endpoints["summarize"] = "POST /ai/summarize"
    if settings.ENABLE_TOKEN_COUNTING:
        ai_endpoints.update({
            "count_tokens": "POST /ai/tokens/count",
            "batch_count_tokens": "POST /ai/tokens/batch",
        })
    
    if ai_endpoints:
        endpoints["ai_service_endpoints"] = ai_endpoints
    
    return endpoints


# Uvicorn launch (optional if using gunicorn/uvicorn CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
