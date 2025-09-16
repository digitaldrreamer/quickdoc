from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PAGE_BASED = "page_based"

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC,
        description="Strategy for chunking the document text"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )
    max_chunk_size: Optional[int] = Field(
        default=None,
        description="Maximum characters per chunk (overrides default)"
    )
    min_chunk_size: Optional[int] = Field(
        default=None,
        description="Minimum characters per chunk (overrides default)"
    )
    overlap_size: Optional[int] = Field(
        default=None,
        description="Character overlap between chunks (overrides default)"
    )

class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    page_number: int = Field(..., description="Page number where chunk originates")
    chunk_index: int = Field(..., description="Index of chunk within the page")
    char_count: int = Field(..., description="Number of characters in the chunk")
    word_count: int = Field(..., description="Number of words in the chunk")
    contains_headers: bool = Field(..., description="Whether chunk contains headers")
    contains_footnotes: bool = Field(..., description="Whether chunk contains footnotes")
    semantic_boundary: str = Field(..., description="Type of semantic boundary")
    start_char: int = Field(..., description="Starting character position in document")
    end_char: int = Field(..., description="Ending character position in document")

class EmbeddingChunk(BaseModel):
    """A single chunk with its embedding."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding of the text")
    metadata: ChunkMetadata = Field(..., description="Metadata about the chunk")

class EmbeddingStats(BaseModel):
    """Statistics about the embedding generation process."""
    total_chunks: int = Field(..., description="Total number of chunks created")
    total_pages: int = Field(..., description="Total number of pages processed")
    avg_chunk_size_chars: float = Field(..., description="Average chunk size in characters")
    min_chunk_size_chars: int = Field(..., description="Minimum chunk size in characters")
    max_chunk_size_chars: int = Field(..., description="Maximum chunk size in characters")
    avg_chunk_size_words: float = Field(..., description="Average chunk size in words")
    min_chunk_size_words: int = Field(..., description="Minimum chunk size in words")
    max_chunk_size_words: int = Field(..., description="Maximum chunk size in words")
    chunks_with_headers: int = Field(..., description="Number of chunks containing headers")
    chunks_with_footnotes: int = Field(..., description="Number of chunks containing footnotes")
    embedding_dimensions: int = Field(..., description="Dimension of the embedding vectors")
    avg_embedding_norm: float = Field(..., description="Average norm of embedding vectors")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    success: bool = Field(..., description="Whether the operation was successful")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension")
    chunks: List[EmbeddingChunk] = Field(..., description="List of chunks with embeddings")
    stats: EmbeddingStats = Field(..., description="Statistics about the process")
    processing_method: str = Field(..., description="Method used for text extraction")
    chunking_strategy: str = Field(..., description="Strategy used for chunking")
    metrics: Dict[str, Any] = Field(..., description="Additional processing metrics")

class EmbeddingErrorResponse(BaseModel):
    """Error response model for embedding generation."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    filename: Optional[str] = Field(default=None, description="Filename if available")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""
    files: List[str] = Field(..., description="List of file paths to process")
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC,
        description="Strategy for chunking the document text"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )
    max_chunk_size: Optional[int] = Field(
        default=None,
        description="Maximum characters per chunk (overrides default)"
    )
    min_chunk_size: Optional[int] = Field(
        default=None,
        description="Minimum characters per chunk (overrides default)"
    )
    overlap_size: Optional[int] = Field(
        default=None,
        description="Character overlap between chunks (overrides default)"
    )

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    success: bool = Field(..., description="Whether the operation was successful")
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(..., description="Number of successfully processed files")
    failed_files: int = Field(..., description="Number of failed files")
    results: List[EmbeddingResponse] = Field(..., description="Results for successful files")
    errors: List[EmbeddingErrorResponse] = Field(..., description="Errors for failed files")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    total_chunks: int = Field(..., description="Total number of chunks across all files")
    total_embeddings: int = Field(..., description="Total number of embeddings generated")
