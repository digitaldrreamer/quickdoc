import logging
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Different chunking strategies for different use cases."""
    SEMANTIC = "semantic"  # Split by semantic boundaries (paragraphs, sections)
    FIXED_SIZE = "fixed_size"  # Split by fixed character count
    SENTENCE_BASED = "sentence_based"  # Split by sentences
    PAGE_BASED = "page_based"  # Keep page boundaries intact

@dataclass
class ChunkMetadata:
    """Metadata for each text chunk."""
    chunk_id: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    char_count: int
    contains_headers: bool
    contains_footnotes: bool
    semantic_boundary: str  # paragraph, section, page, etc.

@dataclass
class ChunkedText:
    """A chunk of text with its metadata."""
    text: str
    metadata: ChunkMetadata

class SmartChunkingService:
    """
    Service for intelligently chunking text from PDFs and EPUBs.
    
    This service creates chunks optimized for:
    1. AI consumption (maintaining semantic coherence)
    2. Search relevance (preserving context and structure)
    3. Retrieval accuracy (appropriate chunk sizes)
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 200,
                 overlap_size: int = 100,
                 strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC):
        """
        Initialize the chunking service.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            overlap_size: Character overlap between chunks for context preservation
            strategy: Chunking strategy to use
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.strategy = strategy
        
        # * Common patterns for semantic boundaries
        self.section_patterns = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            r'^\d+\.\s+',  # Numbered sections
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
        ]
        
        logger.info(f"Initialized SmartChunkingService with strategy: {strategy.value}")
    
    def chunk_pages(self, pages: List[str], filename: str) -> List[ChunkedText]:
        """
        Chunk a list of page texts into optimal chunks.
        
        Args:
            pages: List of page texts
            filename: Original filename for metadata
            
        Returns:
            List of ChunkedText objects
        """
        all_chunks = []
        global_char_offset = 0
        
        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue
                
            # * Process each page
            page_chunks = self._chunk_single_page(
                page_text, page_num, global_char_offset, filename
            )
            all_chunks.extend(page_chunks)
            
            # * Update global character offset
            global_char_offset += len(page_text)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def chunk_chapters(self, chapters: List[str], filename: str) -> List[ChunkedText]:
        """
        Chunk a list of chapter texts into optimal chunks.
        
        Args:
            chapters: List of chapter texts
            filename: Original filename for metadata
            
        Returns:
            List of ChunkedText objects
        """
        all_chunks = []
        global_char_offset = 0
        
        for chapter_num, chapter_text in enumerate(chapters, 1):
            if not chapter_text.strip():
                continue
                
            # * Process each chapter
            chapter_chunks = self._chunk_single_chapter(
                chapter_text, chapter_num, global_char_offset, filename
            )
            all_chunks.extend(chapter_chunks)
            
            # * Update global character offset
            global_char_offset += len(chapter_text)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(chapters)} chapters")
        return all_chunks
    
    def _chunk_single_page(self, page_text: str, page_num: int, 
                          global_offset: int, filename: str) -> List[ChunkedText]:
        """Chunk a single page of text."""
        if self.strategy == ChunkingStrategy.PAGE_BASED:
            return self._create_page_based_chunk(page_text, page_num, global_offset, filename)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._create_semantic_chunks(page_text, page_num, global_offset, filename)
        elif self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._create_fixed_size_chunks(page_text, page_num, global_offset, filename)
        elif self.strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._create_sentence_chunks(page_text, page_num, global_offset, filename)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_single_chapter(self, chapter_text: str, chapter_num: int, 
                             global_offset: int, filename: str) -> List[ChunkedText]:
        """Chunk a single chapter of text."""
        # * For chapters, we treat them like pages but with chapter metadata
        return self._chunk_single_page(chapter_text, chapter_num, global_offset, filename)
    
    def _create_page_based_chunk(self, page_text: str, page_num: int, 
                               global_offset: int, filename: str) -> List[ChunkedText]:
        """Create a single chunk for the entire page."""
        chunk_id = f"{filename}_page_{page_num}_chunk_0"
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            page_number=page_num,
            chunk_index=0,
            start_char=global_offset,
            end_char=global_offset + len(page_text),
            word_count=len(page_text.split()),
            char_count=len(page_text),
            contains_headers=self._has_headers(page_text),
            contains_footnotes=self._has_footnotes(page_text),
            semantic_boundary="page"
        )
        
        return [ChunkedText(text=page_text.strip(), metadata=metadata)]
    
    def _create_semantic_chunks(self, page_text: str, page_num: int, 
                              global_offset: int, filename: str) -> List[ChunkedText]:
        """Create chunks based on semantic boundaries (paragraphs, sections)."""
        chunks = []
        
        # * Split by paragraphs first
        paragraphs = self._split_by_paragraphs(page_text)
        
        current_chunk = ""
        chunk_index = 0
        char_offset = global_offset
        
        for paragraph in paragraphs:
            # * Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # * Create chunk from current content
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk_id = f"{filename}_page_{page_num}_chunk_{chunk_index}"
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        end_char=char_offset + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        contains_headers=self._has_headers(current_chunk),
                        contains_footnotes=self._has_footnotes(current_chunk),
                        semantic_boundary="paragraph"
                    )
                    
                    chunks.append(ChunkedText(text=current_chunk.strip(), metadata=metadata))
                    chunk_index += 1
                
                # * Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                char_offset += len(current_chunk) - len(overlap_text)
            else:
                current_chunk += paragraph + "\n\n"
        
        # * Add final chunk if it has content
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_id = f"{filename}_page_{page_num}_chunk_{chunk_index}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                page_number=page_num,
                chunk_index=chunk_index,
                start_char=char_offset,
                end_char=char_offset + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                contains_headers=self._has_headers(current_chunk),
                contains_footnotes=self._has_footnotes(current_chunk),
                semantic_boundary="paragraph"
            )
            
            chunks.append(ChunkedText(text=current_chunk.strip(), metadata=metadata))
        
        return chunks
    
    def _create_fixed_size_chunks(self, page_text: str, page_num: int, 
                                 global_offset: int, filename: str) -> List[ChunkedText]:
        """Create chunks of fixed size with overlap."""
        chunks = []
        chunk_index = 0
        char_offset = global_offset
        
        while char_offset < global_offset + len(page_text):
            # * Calculate chunk boundaries
            chunk_start = char_offset - global_offset
            chunk_end = min(chunk_start + self.max_chunk_size, len(page_text))
            
            # * Extract chunk text
            chunk_text = page_text[chunk_start:chunk_end]
            
            # * Try to end at a sentence boundary
            if chunk_end < len(page_text):
                chunk_text = self._adjust_to_sentence_boundary(chunk_text, page_text, chunk_start)
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_id = f"{filename}_page_{page_num}_chunk_{chunk_index}"
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    contains_headers=self._has_headers(chunk_text),
                    contains_footnotes=self._has_footnotes(chunk_text),
                    semantic_boundary="fixed_size"
                )
                
                chunks.append(ChunkedText(text=chunk_text.strip(), metadata=metadata))
                chunk_index += 1
            
            # * Move to next chunk with overlap
            char_offset += len(chunk_text) - self.overlap_size
        
        return chunks
    
    def _create_sentence_chunks(self, page_text: str, page_num: int, 
                               global_offset: int, filename: str) -> List[ChunkedText]:
        """Create chunks based on sentence boundaries."""
        chunks = []
        
        # * Split into sentences
        sentences = self._split_by_sentences(page_text)
        
        current_chunk = ""
        chunk_index = 0
        char_offset = global_offset
        
        for sentence in sentences:
            # * Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # * Create chunk from current content
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk_id = f"{filename}_page_{page_num}_chunk_{chunk_index}"
                    
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        end_char=char_offset + len(current_chunk),
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        contains_headers=self._has_headers(current_chunk),
                        contains_footnotes=self._has_footnotes(current_chunk),
                        semantic_boundary="sentence"
                    )
                    
                    chunks.append(ChunkedText(text=current_chunk.strip(), metadata=metadata))
                    chunk_index += 1
                
                # * Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                char_offset += len(current_chunk) - len(overlap_text)
            else:
                current_chunk += sentence + " "
        
        # * Add final chunk if it has content
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_id = f"{filename}_page_{page_num}_chunk_{chunk_index}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                page_number=page_num,
                chunk_index=chunk_index,
                start_char=char_offset,
                end_char=char_offset + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                contains_headers=self._has_headers(current_chunk),
                contains_footnotes=self._has_footnotes(current_chunk),
                semantic_boundary="sentence"
            )
            
            chunks.append(ChunkedText(text=current_chunk.strip(), metadata=metadata))
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        # * Split by double newlines or paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        # * Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _adjust_to_sentence_boundary(self, chunk_text: str, full_text: str, start_pos: int) -> str:
        """Adjust chunk to end at a sentence boundary."""
        # * Find the last sentence ending in the chunk
        last_period = chunk_text.rfind('.')
        last_exclamation = chunk_text.rfind('!')
        last_question = chunk_text.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > self.min_chunk_size:
            return chunk_text[:last_sentence_end + 1]
        
        return chunk_text
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of the current chunk."""
        if len(text) <= self.overlap_size:
            return text
        
        # * Try to get overlap at sentence boundary
        overlap_start = len(text) - self.overlap_size
        overlap_text = text[overlap_start:]
        
        # * Find first sentence start in overlap
        first_sentence_start = overlap_text.find('. ') + 2
        if first_sentence_start > 1:
            return overlap_text[first_sentence_start:]
        
        return overlap_text
    
    def _has_headers(self, text: str) -> bool:
        """Check if text contains headers."""
        for pattern in self.section_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _has_footnotes(self, text: str) -> bool:
        """Check if text contains footnotes."""
        footnote_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'^\d+\.\s+',  # Numbered footnotes
            r'^Footnote\s+\d+',  # Footnote 1, etc.
        ]
        
        for pattern in footnote_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def get_chunking_stats(self, chunks: List[ChunkedText]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        if not chunks:
            return {}
        
        char_counts = [chunk.metadata.char_count for chunk in chunks]
        word_counts = [chunk.metadata.word_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_chars": sum(char_counts) / len(char_counts),
            "min_chunk_size_chars": min(char_counts),
            "max_chunk_size_chars": max(char_counts),
            "avg_chunk_size_words": sum(word_counts) / len(word_counts),
            "min_chunk_size_words": min(word_counts),
            "max_chunk_size_words": max(word_counts),
            "chunks_with_headers": sum(1 for chunk in chunks if chunk.metadata.contains_headers),
            "chunks_with_footnotes": sum(1 for chunk in chunks if chunk.metadata.contains_footnotes),
            "semantic_boundaries": {
                boundary: sum(1 for chunk in chunks if chunk.metadata.semantic_boundary == boundary)
                for boundary in set(chunk.metadata.semantic_boundary for chunk in chunks)
            }
        }
