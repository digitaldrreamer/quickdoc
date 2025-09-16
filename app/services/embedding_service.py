import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from ..services.chunking_service import ChunkedText, SmartChunkingService, ChunkingStrategy

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""
    chunk_id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', max_batch_size: int = 32):
        """
        Initialize the embedding service with enhanced capabilities.
        
        Args:
            model_name: Sentence transformer model name
            max_batch_size: Maximum batch size for embedding generation
        """
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimensions = self.model.get_sentence_embedding_dimension()
        self.max_batch_size = max_batch_size
        self.chunking_service = SmartChunkingService()
        
        logger.info(f"Initialized EmbeddingService with model {model_name} on CPU.")
        logger.info(f"Embedding dimensions: {self.dimensions}")
        logger.info(f"Max batch size: {max_batch_size}")

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()

    def embed_document_text(self, text: str, normalize: bool = True) -> List[List[float]]:
        """Legacy method for backward compatibility."""
        # Simple chunking strategy
        sentences = text.split('. ')
        
        # In a more advanced scenario, you'd use a text splitter like RecursiveCharacterTextSplitter
        # For this, we'll just batch encode sentences.
        embeddings = self.model.encode(sentences, normalize_embeddings=normalize, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    async def embed_chunks(self, chunks: List[ChunkedText], 
                          normalize: bool = True, 
                          show_progress: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of text chunks with batch processing.
        
        Args:
            chunks: List of ChunkedText objects
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # * Process in batches to avoid memory issues
        all_results = []
        
        for i in range(0, len(chunks), self.max_batch_size):
            batch = chunks[i:i + self.max_batch_size]
            batch_texts = [chunk.text for chunk in batch]
            
            # * Generate embeddings for this batch
            batch_embeddings = await asyncio.to_thread(
                self._encode_batch, batch_texts, normalize, show_progress
            )
            
            # * Create results with metadata
            for chunk, embedding in zip(batch, batch_embeddings):
                result = EmbeddingResult(
                    chunk_id=chunk.metadata.chunk_id,
                    embedding=embedding.tolist(),
                    text=chunk.text,
                    metadata={
                        "page_number": chunk.metadata.page_number,
                        "chunk_index": chunk.metadata.chunk_index,
                        "char_count": chunk.metadata.char_count,
                        "word_count": chunk.metadata.word_count,
                        "contains_headers": chunk.metadata.contains_headers,
                        "contains_footnotes": chunk.metadata.contains_footnotes,
                        "semantic_boundary": chunk.metadata.semantic_boundary,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char
                    }
                )
                all_results.append(result)
        
        logger.info(f"Generated {len(all_results)} embeddings")
        return all_results

    def _encode_batch(self, texts: List[str], normalize: bool, show_progress: bool) -> np.ndarray:
        """Encode a batch of texts synchronously."""
        return self.model.encode(
            texts, 
            normalize_embeddings=normalize, 
            show_progress_bar=show_progress,
            batch_size=min(len(texts), self.max_batch_size)
        )

    async def embed_pages(self, pages: List[str], filename: str,
                         chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                         normalize: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for pages with intelligent chunking.
        
        Args:
            pages: List of page texts
            filename: Original filename for metadata
            chunking_strategy: Strategy for chunking the text
            normalize: Whether to normalize embeddings
            
        Returns:
            List of EmbeddingResult objects
        """
        logger.info(f"Processing {len(pages)} pages with {chunking_strategy.value} chunking")
        
        # * Create chunking service with specified strategy
        chunking_service = SmartChunkingService(strategy=chunking_strategy)
        
        # * Chunk the pages
        chunks = chunking_service.chunk_pages(pages, filename)
        
        if not chunks:
            logger.warning("No chunks created from pages")
            return []
        
        # * Generate embeddings for chunks
        results = await self.embed_chunks(chunks, normalize)
        
        # * Log chunking statistics
        stats = chunking_service.get_chunking_stats(chunks)
        logger.info(f"Chunking stats: {stats}")
        
        return results

    async def embed_chapters(self, chapters: List[str], filename: str,
                            chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                            normalize: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for chapters with intelligent chunking.
        
        Args:
            chapters: List of chapter texts
            filename: Original filename for metadata
            chunking_strategy: Strategy for chunking the text
            normalize: Whether to normalize embeddings
            
        Returns:
            List of EmbeddingResult objects
        """
        logger.info(f"Processing {len(chapters)} chapters with {chunking_strategy.value} chunking")
        
        # * Create chunking service with specified strategy
        chunking_service = SmartChunkingService(strategy=chunking_strategy)
        
        # * Chunk the chapters
        chunks = chunking_service.chunk_chapters(chapters, filename)
        
        if not chunks:
            logger.warning("No chunks created from chapters")
            return []
        
        # * Generate embeddings for chunks
        results = await self.embed_chunks(chunks, normalize)
        
        # * Log chunking statistics
        stats = chunking_service.get_chunking_stats(chunks)
        logger.info(f"Chunking stats: {stats}")
        
        return results

    def get_embedding_stats(self, results: List[EmbeddingResult]) -> Dict[str, Any]:
        """Get statistics about the embedding generation process."""
        if not results:
            return {}
        
        embedding_dims = [len(result.embedding) for result in results]
        text_lengths = [len(result.text) for result in results]
        
        # * Calculate embedding statistics
        embeddings_array = np.array([result.embedding for result in results])
        embedding_norms = np.linalg.norm(embeddings_array, axis=1)
        
        return {
            "total_embeddings": len(results),
            "embedding_dimensions": embedding_dims[0] if embedding_dims else 0,
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "avg_embedding_norm": float(np.mean(embedding_norms)),
            "min_embedding_norm": float(np.min(embedding_norms)),
            "max_embedding_norm": float(np.max(embedding_norms)),
            "pages_covered": len(set(result.metadata["page_number"] for result in results)),
            "chunks_with_headers": sum(1 for result in results if result.metadata["contains_headers"]),
            "chunks_with_footnotes": sum(1 for result in results if result.metadata["contains_footnotes"])
        }