import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse

from ..services.embedding_service import EmbeddingService, EmbeddingResult
from ..services.chunking_service import ChunkingStrategy
from ..processor import (
    extract_text_by_pages_from_pdf, 
    extract_text_by_chapters_from_epub,
    extract_text_from_doc
)
from ..metrics import ResourceTracker
from ..models.embedding_models import (
    EmbeddingRequest, EmbeddingResponse, EmbeddingErrorResponse,
    EmbeddingChunk, ChunkMetadata, EmbeddingStats,
    BatchEmbeddingRequest, BatchEmbeddingResponse
)
from ..core.config import settings

logger = logging.getLogger(__name__)

# * Create router for embedding endpoints
router = APIRouter(prefix="/embed", tags=["Document Embeddings"])

def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service instance."""
    if not settings.ENABLE_EMBEDDING_MODEL:
        raise HTTPException(
            status_code=503, 
            detail="Embedding service is disabled. Set ENABLE_EMBEDDING_MODEL=true to enable."
        )
    
    try:
        return EmbeddingService()
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize embedding service"
        )

@router.post("/document", response_model=EmbeddingResponse)
async def embed_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(default="semantic"),
    normalize: bool = Form(default=True),
    max_chunk_size: int = Form(default=None),
    min_chunk_size: int = Form(default=None),
    overlap_size: int = Form(default=None),
    embedder: EmbeddingService = Depends(get_embedding_service)
):
    """
    Convert PDF/EPUB document to page-by-page embeddings with intelligent chunking.
    
    This endpoint processes documents and creates embeddings optimized for:
    - AI consumption (maintaining semantic coherence)
    - Search relevance (preserving context and structure)
    - Retrieval accuracy (appropriate chunk sizes)
    
    Args:
        file: PDF or EPUB file to process
        chunking_strategy: Strategy for chunking ('semantic', 'fixed_size', 'sentence_based', 'page_based')
        normalize: Whether to normalize embeddings
        max_chunk_size: Maximum characters per chunk (overrides default)
        min_chunk_size: Minimum characters per chunk (overrides default)
        overlap_size: Character overlap between chunks (overrides default)
        
    Returns:
        EmbeddingResponse with chunks, embeddings, and metadata
    """
    start_time = time.time()
    tracker = ResourceTracker()
    
    try:
        # * Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.epub']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Only PDF and EPUB are supported."
            )
        
        # * Validate chunking strategy
        try:
            strategy = ChunkingStrategy(chunking_strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunking strategy: {chunking_strategy}. "
                       f"Valid options: {[s.value for s in ChunkingStrategy]}"
            )
        
        # * Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            logger.info(f"Processing {file.filename} with {strategy.value} chunking")
            
            # * Extract text based on file type
            if file_ext == '.pdf':
                if not settings.ENABLE_PDF_PROCESSING:
                    raise HTTPException(
                        status_code=503,
                        detail="PDF processing is disabled. Set ENABLE_PDF_PROCESSING=true to enable."
                    )
                
                # * Extract pages from PDF
                pages = await extract_text_by_pages_from_pdf(temp_file_path, tracker)
                processing_method = tracker.get_method()
                
                if not pages or not any(page.strip() for page in pages):
                    raise HTTPException(
                        status_code=422,
                        detail="No text could be extracted from the PDF"
                    )
                
                # * Generate embeddings for pages
                embedding_results = await embedder.embed_pages(
                    pages, file.filename, strategy, normalize
                )
                
            elif file_ext == '.epub':
                if not settings.ENABLE_EPUB_PROCESSING:
                    raise HTTPException(
                        status_code=503,
                        detail="EPUB processing is disabled. Set ENABLE_EPUB_PROCESSING=true to enable."
                    )
                
                # * Extract chapters from EPUB
                chapters = await extract_text_by_chapters_from_epub(temp_file_path, tracker)
                processing_method = tracker.get_method()
                
                if not chapters or not any(chapter.strip() for chapter in chapters):
                    raise HTTPException(
                        status_code=422,
                        detail="No text could be extracted from the EPUB"
                    )
                
                # * Generate embeddings for chapters
                embedding_results = await embedder.embed_chapters(
                    chapters, file.filename, strategy, normalize
                )
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}"
                )
            
            if not embedding_results:
                raise HTTPException(
                    status_code=422,
                    detail="No embeddings could be generated from the document"
                )
            
            # * Convert results to response format
            chunks = []
            for result in embedding_results:
                chunk_metadata = ChunkMetadata(
                    chunk_id=result.chunk_id,
                    page_number=result.metadata["page_number"],
                    chunk_index=result.metadata["chunk_index"],
                    char_count=result.metadata["char_count"],
                    word_count=result.metadata["word_count"],
                    contains_headers=result.metadata["contains_headers"],
                    contains_footnotes=result.metadata["contains_footnotes"],
                    semantic_boundary=result.metadata["semantic_boundary"],
                    start_char=result.metadata["start_char"],
                    end_char=result.metadata["end_char"]
                )
                
                chunk = EmbeddingChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    embedding=result.embedding,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
            
            # * Generate statistics
            embedding_stats = embedder.get_embedding_stats(embedding_results)
            processing_time = (time.time() - start_time) * 1000
            
            stats = EmbeddingStats(
                total_chunks=len(chunks),
                total_pages=len(pages) if file_ext == '.pdf' else len(chapters),
                avg_chunk_size_chars=embedding_stats.get("avg_text_length", 0),
                min_chunk_size_chars=embedding_stats.get("min_text_length", 0),
                max_chunk_size_chars=embedding_stats.get("max_text_length", 0),
                avg_chunk_size_words=sum(chunk.metadata.word_count for chunk in chunks) / len(chunks) if chunks else 0,
                min_chunk_size_words=min(chunk.metadata.word_count for chunk in chunks) if chunks else 0,
                max_chunk_size_words=max(chunk.metadata.word_count for chunk in chunks) if chunks else 0,
                chunks_with_headers=embedding_stats.get("chunks_with_headers", 0),
                chunks_with_footnotes=embedding_stats.get("chunks_with_footnotes", 0),
                embedding_dimensions=embedding_stats.get("embedding_dimensions", 0),
                avg_embedding_norm=embedding_stats.get("avg_embedding_norm", 0),
                processing_time_ms=processing_time
            )
            
            # * Prepare metrics
            metrics = {
                "processing_duration_ms": processing_time,
                "memory_usage_mb": tracker.get_memory_usage(),
                "processing_method": processing_method,
                "file_size_bytes": len(content),
                "total_characters": sum(len(page) for page in pages) if file_ext == '.pdf' else sum(len(chapter) for chapter in chapters),
                "embedding_model": embedder.model.get_sentence_embedding_dimension(),
                "chunking_parameters": {
                    "strategy": strategy.value,
                    "normalize": normalize,
                    "max_chunk_size": max_chunk_size,
                    "min_chunk_size": min_chunk_size,
                    "overlap_size": overlap_size
                }
            }
            
            response = EmbeddingResponse(
                success=True,
                filename=file.filename,
                file_type=file_ext,
                chunks=chunks,
                stats=stats,
                processing_method=processing_method,
                chunking_strategy=strategy.value,
                metrics=metrics
            )
            
            logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks, {processing_time:.2f}ms")
            return response
            
        finally:
            # * Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/text", response_model=Dict[str, Any])
async def embed_text_chunks(
    text: str = Form(...),
    chunking_strategy: str = Form(default="semantic"),
    normalize: bool = Form(default=True),
    max_chunk_size: int = Form(default=1000),
    min_chunk_size: int = Form(default=200),
    overlap_size: int = Form(default=100),
    embedder: EmbeddingService = Depends(get_embedding_service)
):
    """
    Generate embeddings for text with intelligent chunking.
    
    Args:
        text: Text to process and embed
        chunking_strategy: Strategy for chunking
        normalize: Whether to normalize embeddings
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap_size: Character overlap between chunks
        
    Returns:
        Dictionary with chunks and embeddings
    """
    try:
        # * Validate chunking strategy
        try:
            strategy = ChunkingStrategy(chunking_strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunking strategy: {chunking_strategy}"
            )
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # * Create chunking service with custom parameters
        from ..services.chunking_service import SmartChunkingService
        chunking_service = SmartChunkingService(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size,
            strategy=strategy
        )
        
        # * Split text into pages (treat as single page for now)
        pages = [text]
        
        # * Generate chunks
        chunks = chunking_service.chunk_pages(pages, "text_input")
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks could be created from text")
        
        # * Generate embeddings
        embedding_results = await embedder.embed_chunks(chunks, normalize)
        
        # * Convert to response format
        result_chunks = []
        for result in embedding_results:
            chunk_data = {
                "chunk_id": result.chunk_id,
                "text": result.text,
                "embedding": result.embedding,
                "metadata": result.metadata
            }
            result_chunks.append(chunk_data)
        
        # * Generate statistics
        stats = embedder.get_embedding_stats(embedding_results)
        chunking_stats = chunking_service.get_chunking_stats(chunks)
        
        response = {
            "success": True,
            "total_chunks": len(result_chunks),
            "chunks": result_chunks,
            "embedding_stats": stats,
            "chunking_stats": chunking_stats,
            "chunking_strategy": strategy.value,
            "parameters": {
                "max_chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size,
                "overlap_size": overlap_size,
                "normalize": normalize
            }
        }
        
        logger.info(f"Successfully processed text: {len(result_chunks)} chunks")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/strategies")
async def get_chunking_strategies():
    """Get available chunking strategies with descriptions."""
    strategies = {
        "semantic": {
            "name": "Semantic Chunking",
            "description": "Split by semantic boundaries (paragraphs, sections) for optimal context preservation",
            "best_for": "AI consumption, search relevance, maintaining meaning"
        },
        "fixed_size": {
            "name": "Fixed Size Chunking", 
            "description": "Split by fixed character count with sentence boundary adjustment",
            "best_for": "Consistent chunk sizes, processing efficiency"
        },
        "sentence_based": {
            "name": "Sentence-Based Chunking",
            "description": "Split by sentence boundaries with overlap for context",
            "best_for": "Natural language processing, maintaining sentence integrity"
        },
        "page_based": {
            "name": "Page-Based Chunking",
            "description": "Keep entire pages as single chunks",
            "best_for": "Page-level retrieval, maintaining page context"
        }
    }
    
    return {
        "available_strategies": list(strategies.keys()),
        "strategy_details": strategies
    }

@router.get("/health")
async def embedding_health_check():
    """Health check for embedding service."""
    try:
        if not settings.ENABLE_EMBEDDING_MODEL:
            return {
                "status": "disabled",
                "message": "Embedding service is disabled",
                "enabled_features": []
            }
        
        # * Try to initialize service
        embedder = EmbeddingService()
        
        return {
            "status": "healthy",
            "message": "Embedding service is operational",
            "enabled_features": [
                "document_embedding",
                "text_embedding", 
                "chunking_strategies",
                "batch_processing"
            ],
            "model_info": {
                "dimensions": embedder.dimensions,
                "max_batch_size": embedder.max_batch_size
            }
        }
        
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Embedding service error: {str(e)}",
            "enabled_features": []
        }
