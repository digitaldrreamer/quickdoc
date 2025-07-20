import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict
import asyncio

logger = logging.getLogger(__name__)

class OptimizedSummarizer:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.device = "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16  # Optional: useful if your CPU supports it
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_chunk_length = self.tokenizer.model_max_length - 2  # Buffer for special tokens
        logger.info(f"Initialized OptimizedSummarizer with model {model_name} on {self.device}")

    async def summarize(self, text: str, max_length: int = 120, min_length: int = 30, quality: str = "high") -> Dict[str, any]:
        return await asyncio.to_thread(
            self._summarize_sync, text, max_length, min_length, quality
        )

    def _summarize_sync(self, text: str, max_length: int, min_length: int, quality: str) -> Dict[str, any]:
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]

        if len(tokens) <= self.max_chunk_length:
            summary = self._generate_summary(text, max_length, min_length, quality)
        else:
            logger.info(f"Text too long ({len(tokens)} tokens), splitting into chunks...")
            chunks = self._create_chunks(tokens)

            chunk_summaries = [
                self._generate_summary(
                    self.tokenizer.decode(chunk, skip_special_tokens=True),
                    max_len=80,
                    min_len=20,
                    quality=quality
                )
                for chunk in chunks
            ]
            combined_summary_text = " ".join(chunk_summaries)
            summary = self._generate_summary(combined_summary_text, max_length, min_length, quality)

        return {
            "summary": summary,
            "original_char_count": len(text),
            "summary_char_count": len(summary)
        }

    def _create_chunks(self, tokens: torch.Tensor):
        overlap = 100
        step = self.max_chunk_length - overlap
        chunks = [tokens[i:i + self.max_chunk_length] for i in range(0, len(tokens), step)]
        return chunks

    def _generate_summary(self, text_to_summarize: str, max_len: int, min_len: int, quality: str) -> str:
        # T5-style prompt
        prompt = f"summarize: {text_to_summarize.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_chunk_length, truncation=True).to(self.device)

        beam_search_params = {
            "num_beams": 4 if quality == "high" else 1,
            "early_stopping": True,
            "length_penalty": 2.0,
        }

        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_len,
            min_length=min_len,
            **beam_search_params
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
