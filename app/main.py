from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from contextlib import asynccontextmanager
import asyncio

from .routes import router as doc_conversion_router
from .api.ai_routes import router as ai_router
from .services.embedding_service import EmbeddingService
from .services.summarization_service import OptimizedSummarizer
from .services.token_service import TokenCounter
from .core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)


# Lifespan manager to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading AI models...")
    loop = asyncio.get_event_loop()

    embedding_service_future = loop.run_in_executor(None, EmbeddingService)
    summarizer_future = loop.run_in_executor(None, OptimizedSummarizer)
    token_counter_future = loop.run_in_executor(None, TokenCounter)

    app.state.embedder, app.state.summarizer, app.state.tokenizer = await asyncio.gather(
        embedding_service_future,
        summarizer_future,
        token_counter_future
    )

    logger.info("All AI models loaded.")

    app.state.summarization_queue = asyncio.Queue()
    app.state.summarization_worker = asyncio.create_task(
        summarization_worker(app.state.summarization_queue, app.state.summarizer)
    )
    logger.info("Summarization worker started.")

    yield

    app.state.summarization_worker.cancel()
    logger.info("Summarization worker stopped.")
    logger.info("Application shutdown.")


# Background worker for summarization queue
async def summarization_worker(queue: asyncio.Queue, summarizer: OptimizedSummarizer):
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
    title="Kuro-Bot Document & AI Processor",
    description="A microservice for text extraction, PDF conversion, embedding, summarization, and tokenization.",
    version="2.0.0",
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
                if origin.endswith(".kuroconnect.com") and origin not in self._known_origins:
                    self.allow_origins.append(origin)
                    self._known_origins.add(origin)
                    logger.info(f"Dynamically allowed CORS origin: {origin}")
        return await super().__call__(scope, receive, send)


# Register CORS middleware
app.add_middleware(
    DynamicCORS,
    allow_origins=[],  # initially empty, filled dynamically
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(doc_conversion_router, tags=["Document Conversion"])
app.include_router(ai_router, prefix="/ai", tags=["AI Services"])


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
    return {"status": "healthy", "service": "kuro-bot-processor"}


# Root
@app.get("/", tags=["Utilities"])
async def root():
    return {
        "message": "Kuro-Bot Document & AI Processor",
        "documentation": "/docs",
        "document_conversion_endpoints": {
            "extract_text": "POST /extract",
            "convert_to_pdf": "POST /convert-to-pdf",
        },
        "ai_service_endpoints": {
            "embed_text": "POST /ai/embed/text",
            "embed_document": "POST /ai/embed/document",
            "summarize": "POST /ai/summarize",
            "count_tokens": "POST /ai/tokens/count",
            "batch_count_tokens": "POST /ai/tokens/batch",
        }
    }


# Uvicorn launch (optional if using gunicorn/uvicorn CLI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
