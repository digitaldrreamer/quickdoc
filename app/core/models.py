from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ModelName(str, Enum):
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    GEMINI = "gemini"

# --- Embedding Models ---
class EmbedTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to be embedded.", json_schema_extra={"format": "textarea"})
    normalize: bool = Field(True, description="Whether to normalize the embedding vectors.")

class EmbedTextResponse(BaseModel):
    embedding: List[float] = Field(..., example=[0.1, 0.2, 0.3, 0.4, 0.5])
    dimensions: int = Field(..., example=384)

class EmbedDocumentResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    chunk_count: int = Field(..., example=2)
    dimensions: int = Field(..., example=384)
    text_char_count: int = Field(..., example=128)
    
# --- Summarization Models ---
class Quality(str, Enum):
    HIGH = "high"
    FAST = "fast"

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Text to be summarized.", json_schema_extra={"format": "textarea"})
    max_length: int = Field(150, gt=20, description="Maximum length of the summary.")
    min_length: int = Field(30, gt=10, description="Minimum length of the summary.")
    quality: Quality = Field(Quality.HIGH, description="Summarization quality (high=beam search, fast=greedy).")

class SummarizeResponse(BaseModel):
    summary: str = Field(..., example="This is a summary of the text.")
    original_char_count: int = Field(..., example=256)
    summary_char_count: int = Field(..., example=64)
    
# --- Tokenizer Models ---
class TokenCountRequest(BaseModel):
    text: str = Field(..., description="Text to count tokens for.", json_schema_extra={"format": "textarea"})

class TokenCountResponse(BaseModel):
    model: ModelName = Field(..., example=ModelName.LLAMA3)
    token_count: int = Field(..., example=8)
    char_count: int = Field(..., example=35)

class TokenBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="A list of texts to count tokens for.", json_schema_extra={"format": "textarea"})

class TokenBatchResponse(BaseModel):
    model: ModelName = Field(..., example=ModelName.LLAMA3)
    total_token_count: int = Field(..., example=15)
    individual_counts: List[int] = Field(..., example=[3, 12])