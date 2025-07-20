from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
import logging
import tempfile
import os
import asyncio
from typing import List

from app.core.models import *
from app.core.config import settings
from app.processor import extract_text_from_doc

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency to get ML models from app state ---
def get_embedder(request: Request):
    return request.app.state.embedder

def get_tokenizer(request: Request):
    return request.app.state.tokenizer

# --- Embedding Endpoints ---

@router.post("/embed/text", response_model=EmbedTextResponse)
async def embed_text(
    request: EmbedTextRequest,
    embedder = Depends(get_embedder)
):
    """Embeds a single string of text."""
    embedding = await asyncio.to_thread(
        embedder.embed_text, request.text, request.normalize
    )
    return EmbedTextResponse(embedding=embedding, dimensions=embedder.dimensions)

@router.post("/embed/document", response_model=EmbedDocumentResponse)
async def embed_document(
    file: UploadFile = File(...),
    embedder = Depends(get_embedder)
):
    """Extracts text from a document and returns embeddings for text chunks."""
    if file.size > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Limit is {settings.MAX_FILE_SIZE_MB}MB."
        )

    temp_file_path = ""
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        extracted_text = await extract_text_from_doc(temp_file_path, file.filename)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        embeddings = await asyncio.to_thread(
            embedder.embed_document_text, extracted_text
        )

        return EmbedDocumentResponse(
            embeddings=embeddings,
            chunk_count=len(embeddings),
            dimensions=embedder.dimensions,
            text_char_count=len(extracted_text)
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Summarization Endpoint ---

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: Request, summarize_request: SummarizeRequest):
    """Summarizes a long piece of text using a queued background worker."""
    queue = request.app.state.summarization_queue
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    await queue.put((summarize_request.model_dump(), future))

    try:
        summary = await asyncio.wait_for(future, timeout=300) # 5 minute timeout
        return SummarizeResponse(
            summary=summary,
            original_char_count=len(summarize_request.text),
            summary_char_count=len(summary)
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Summarization request timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Tokenizer Endpoints ---

@router.post("/tokens/count/{model}", response_model=TokenCountResponse)
async def count_tokens(
    model: ModelName,
    request: TokenCountRequest,
    tokenizer = Depends(get_tokenizer)
):
    """Counts tokens for a given text and model."""
    try:
        count = await asyncio.to_thread(
            tokenizer.count_tokens, request.text, model.value
        )
        return TokenCountResponse(
            model=model,
            token_count=count,
            char_count=len(request.text)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/tokens/batch/{model}", response_model=TokenBatchResponse)
async def batch_count_tokens(
    model: ModelName,
    request: TokenBatchRequest,
    tokenizer = Depends(get_tokenizer)
):
    """Counts tokens for a batch of texts for a given model."""
    try:
        counts = await asyncio.to_thread(
            tokenizer.batch_count_tokens, request.texts, model.value
        )
        return TokenBatchResponse(
            model=model,
            total_token_count=sum(counts),
            individual_counts=counts
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))