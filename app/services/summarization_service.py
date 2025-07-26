import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, List
import asyncio
import math
from ..core.config import settings

logger = logging.getLogger(__name__)

class OptimizedSummarizer:
    def __init__(self, model_name: str = None):
        """
        Initialize the summarizer with configurable model.
        
        Args:
            model_name: Hugging Face model name. Defaults to config value.
        """
        self.model_name = model_name or settings.SUMMARIZATION_MODEL
        self.device = "cpu"
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # ! Memory optimization for CPU
                low_cpu_mem_usage=True  # ! Additional memory optimization
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # * Calculate effective chunk length with buffer for special tokens
            self.max_chunk_length = max(512, self.tokenizer.model_max_length - 50)  # Safe buffer
            
            logger.info(f"Initialized OptimizedSummarizer with model {self.model_name} on {self.device}")
            logger.info(f"Max chunk length set to {self.max_chunk_length} tokens")
            
        except Exception as e:
            logger.error(f"Failed to initialize summarization model {self.model_name}: {e}")
            raise

    async def summarize(self, text: str, max_length: int = 120, min_length: int = 30, quality: str = "high") -> Dict[str, any]:
        """
        Summarize text with improved chunking and quality handling.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length  
            quality: Quality level ('high', 'medium', 'fast')
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            return await asyncio.to_thread(
                self._summarize_sync, text, max_length, min_length, quality
            )
        except Exception as e:
            logger.error(f"Async summarization failed: {e}")
            raise

    def _summarize_sync(self, text: str, max_length: int, min_length: int, quality: str) -> Dict[str, any]:
        """Synchronous summarization with improved logic."""
        
        if not text or not text.strip():
            return {
                "summary": "",
                "original_char_count": 0,
                "summary_char_count": 0,
                "processing_method": "empty_input"
            }
        
        text = text.strip()
        original_char_count = len(text)
        
        # * Quick return for very short texts
        if original_char_count < 100:
            logger.info("Text too short for summarization, returning original")
            return {
                "summary": text,
                "original_char_count": original_char_count,
                "summary_char_count": original_char_count,
                "processing_method": "passthrough"
            }
        
        # * Tokenize and determine processing strategy
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
        token_count = len(tokens)
        
        logger.info(f"Processing text: {original_char_count} chars, {token_count} tokens")
        
        if token_count <= self.max_chunk_length:
            # * Single chunk processing
            summary = self._generate_summary(text, max_length, min_length, quality)
            processing_method = "single_chunk"
        else:
            # * Multi-chunk processing with improved strategy
            logger.info(f"Text exceeds max length ({token_count} > {self.max_chunk_length}), using multi-chunk processing")
            summary = self._process_long_text(text, tokens, max_length, min_length, quality)
            processing_method = "multi_chunk"

        summary_char_count = len(summary)
        
        # * Validate summary quality
        if summary_char_count == 0:
            logger.warning("Generated empty summary, using text preview")
            summary = text[:max_length * 4]  # Approximate character expansion
            processing_method += "_fallback"
        
        return {
            "summary": summary,
            "original_char_count": original_char_count,
            "summary_char_count": summary_char_count,
            "processing_method": processing_method,
            "compression_ratio": round(original_char_count / summary_char_count, 2) if summary_char_count > 0 else 0
        }

    def _process_long_text(self, text: str, tokens: torch.Tensor, max_length: int, min_length: int, quality: str) -> str:
        """
        Process long text with improved chunking strategy.
        
        Args:
            text: Original text
            tokens: Tokenized text
            max_length: Maximum summary length
            min_length: Minimum summary length
            quality: Quality setting
            
        Returns:
            Final summary
        """
        try:
            # * Create overlapping chunks for better context preservation
            chunks = self._create_smart_chunks(tokens, text)
            logger.info(f"Created {len(chunks)} chunks for processing")
            
            # * Calculate per-chunk summary parameters
            chunk_max_length = max(40, max_length // max(1, len(chunks) - 1))  # Slightly longer per chunk
            chunk_min_length = max(20, min_length // max(1, len(chunks) - 1))
            
            # * Summarize each chunk
            chunk_summaries = []
            for i, chunk_text in enumerate(chunks):
                try:
                    chunk_summary = self._generate_summary(
                        chunk_text,
                        max_len=chunk_max_length,
                        min_len=chunk_min_length,
                        quality=quality
                    )
                    if chunk_summary and chunk_summary.strip():
                        chunk_summaries.append(chunk_summary.strip())
                        logger.debug(f"Chunk {i+1}/{len(chunks)}: {len(chunk_summary)} chars")
                    else:
                        logger.warning(f"Empty summary for chunk {i+1}")
                except Exception as e:
                    logger.error(f"Failed to summarize chunk {i+1}: {e}")
                    continue
            
            if not chunk_summaries:
                logger.error("All chunk summaries failed")
                return text[:max_length * 4]  # Fallback
            
            # * Combine and final summarization
            combined_text = " ".join(chunk_summaries)
            logger.info(f"Combined chunk summaries: {len(combined_text)} chars")
            
            # * Final summarization if combined text is still too long
            combined_tokens = self.tokenizer(combined_text, return_tensors='pt', truncation=False)['input_ids'][0]
            if len(combined_tokens) > self.max_chunk_length:
                logger.info("Combined summary still too long, applying final summarization")
                final_summary = self._generate_summary(combined_text, max_length, min_length, quality)
            else:
                final_summary = self._generate_summary(combined_text, max_length, min_length, quality)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Multi-chunk processing failed: {e}")
            # * Fallback to truncation-based summarization
            truncated_text = self.tokenizer.decode(tokens[:self.max_chunk_length], skip_special_tokens=True)
            return self._generate_summary(truncated_text, max_length, min_length, quality)

    def _create_smart_chunks(self, tokens: torch.Tensor, original_text: str) -> List[str]:
        """
        Create overlapping chunks with sentence boundary awareness.
        
        Args:
            tokens: Tokenized text
            original_text: Original text for sentence splitting
            
        Returns:
            List of text chunks
        """
        # * Calculate chunk parameters
        chunk_size = self.max_chunk_length
        overlap = min(100, chunk_size // 4)  # 25% overlap
        step_size = chunk_size - overlap
        
        # * Simple sentence-aware chunking fallback
        sentences = [s.strip() for s in original_text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            # * For short texts, use token-based chunking
            chunks = []
            for i in range(0, len(tokens), step_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            return chunks
        
        # * Sentence-aware chunking for longer texts
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # * Current chunk is full, start new one
                chunks.append(current_chunk.strip())
                # * Keep some overlap
                if len(chunks) > 1:
                    current_chunk = sentence + ". "
                    current_tokens = sentence_tokens
                else:
                    current_chunk = ""
                    current_tokens = 0
            
            current_chunk += sentence + ". "
            current_tokens += sentence_tokens
        
        # * Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # * Fallback to token chunking if sentence chunking failed
        if not chunks:
            for i in range(0, len(tokens), step_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
        
        return chunks

    def _generate_summary(self, text_to_summarize: str, max_len: int, min_len: int, quality: str) -> str:
        """
        Generate summary with quality-based parameters.
        
        Args:
            text_to_summarize: Text to summarize
            max_len: Maximum length
            min_len: Minimum length
            quality: Quality setting
            
        Returns:
            Generated summary
        """
        try:
            # * Prepare model input with appropriate prompt
            if "t5" in self.model_name.lower():
                prompt = f"summarize: {text_to_summarize.strip()}"
            else:
                prompt = text_to_summarize.strip()
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=self.max_chunk_length, 
                truncation=True
            ).to(self.device)

            # * Quality-based generation parameters
            generation_params = self._get_generation_params(quality, max_len, min_len)
            
            with torch.no_grad():  # ! Memory optimization
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    **generation_params
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # * Clean up the summary
            summary = summary.strip()
            if not summary:
                logger.warning("Model generated empty summary")
                return text_to_summarize[:max_len * 4]  # Character-based fallback
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # * Fallback to simple truncation
            return text_to_summarize[:max_len * 4]

    def _get_generation_params(self, quality: str, max_len: int, min_len: int) -> Dict:
        """Get generation parameters based on quality setting."""
        
        base_params = {
            "max_length": max_len,
            "min_length": min_len,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if quality == "high":
            base_params.update({
                "num_beams": 4,
                "early_stopping": True,
                "length_penalty": 2.0,
                "repetition_penalty": 1.2,
            })
        elif quality == "medium":
            base_params.update({
                "num_beams": 2,
                "early_stopping": True,
                "length_penalty": 1.5,
                "repetition_penalty": 1.1,
            })
        else:  # fast
            base_params.update({
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
            })
        
        return base_params
