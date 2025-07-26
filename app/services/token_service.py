import logging
import os
from transformers import AutoTokenizer
from vertexai.preview import tokenization
from typing import List, Dict
from ..core.config import settings

logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self):
        """Initialize tokenizers for supported models with proper error handling."""
        self.tokenizers = {}
        
        try:
            # * Llama 3 Tokenizer
            logger.info("Loading Llama 3 tokenizer...")
            self.tokenizers["llama3"] = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
            
            # * Mistral Tokenizer (with optional token)
            logger.info("Loading Mistral tokenizer...")
            token = settings.HUGGING_FACE_HUB_TOKEN or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            self.tokenizers["mistral"] = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1", 
                token=token,
                use_auth_token=token if token else None
            )
            
            # * Gemini Tokenizer (requires Google Cloud setup)
            logger.info("Loading Gemini tokenizer...")
            try:
                self.tokenizers["gemini"] = tokenization.get_tokenizer_for_model("gemini-1.5-flash-001")
            except Exception as e:
                logger.warning(f"Failed to load Gemini tokenizer (requires Google Cloud setup): {e}")
                # ! Don't include Gemini if it fails to load
                
            logger.info(f"Initialized TokenCounter with tokenizers: {list(self.tokenizers.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load one or more tokenizers: {e}", exc_info=True)
            # ! Ensure we have at least one working tokenizer
            if not self.tokenizers:
                raise Exception("No tokenizers could be loaded. Please check your configuration and dependencies.")

    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens for a given text and model.
        
        Args:
            text: Input text to tokenize
            model: Model name (must be supported)
            
        Returns:
            Number of tokens
            
        Raises:
            ValueError: If model is not supported
        """
        if model not in self.tokenizers:
            available_models = list(self.tokenizers.keys())
            raise ValueError(f"Model '{model}' not supported. Available models: {available_models}")

        tokenizer = self.tokenizers[model]
        
        try:
            if model == "gemini":
                return tokenizer.count_tokens(text).total_tokens
            else:
                return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception as e:
            logger.error(f"Failed to count tokens for model {model}: {e}")
            raise

    def batch_count_tokens(self, texts: List[str], model: str) -> List[int]:
        """
        Count tokens for a batch of texts for a given model.
        
        Args:
            texts: List of input texts
            model: Model name (must be supported)
            
        Returns:
            List of token counts for each text
            
        Raises:
            ValueError: If model is not supported
        """
        if model not in self.tokenizers:
            available_models = list(self.tokenizers.keys())
            raise ValueError(f"Model '{model}' not supported. Available models: {available_models}")
        
        tokenizer = self.tokenizers[model]
        
        try:
            if model == "gemini":
                return [tokenizer.count_tokens(text).total_tokens for text in texts]
            else:
                # * Use batch encoding for efficiency
                encoded_batch = tokenizer.batch_encode_plus(
                    texts, 
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )
                return [len(tokens) for tokens in encoded_batch['input_ids']]
        except Exception as e:
            logger.error(f"Failed to batch count tokens for model {model}: {e}")
            raise

    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return list(self.tokenizers.keys())