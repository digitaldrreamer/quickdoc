import logging
import os
from transformers import AutoTokenizer
from vertexai.preview import tokenization
from typing import List, Dict

logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self):
        self.tokenizers = {}
        try:
            # Llama 3 Tokenizer
            self.tokenizers["llama3"] = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
            # Mistral Tokenizer
            self.tokenizers["mistral"] = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
            # Gemini Tokenizer (local)
            self.tokenizers["gemini"] = tokenization.get_tokenizer_for_model("gemini-1.5-flash-001")
            logger.info("Initialized TokenCounter with Llama3, Mistral, and Gemini tokenizers.")
        except Exception as e:
            logger.error(f"Failed to load one or more tokenizers: {e}", exc_info=True)
            raise

    def count_tokens(self, text: str, model: str) -> int:
        if model not in self.tokenizers:
            raise ValueError(f"Model '{model}' not supported. Available: {list(self.tokenizers.keys())}")

        tokenizer = self.tokenizers[model]
        
        if model == "gemini":
            return tokenizer.count_tokens(text).total_tokens
        else:
            return len(tokenizer.encode(text))

    def batch_count_tokens(self, texts: List[str], model: str) -> List[int]:
        if model not in self.tokenizers:
            raise ValueError(f"Model '{model}' not supported. Available: {list(self.tokenizers.keys())}")
        
        tokenizer = self.tokenizers[model]

        if model == "gemini":
            return [tokenizer.count_tokens(text).total_tokens for text in texts]
        else:
            return [len(tokens) for tokens in tokenizer.batch_encode_plus(texts)['input_ids']]