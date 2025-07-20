import logging
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimensions = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized EmbeddingService with model {model_name} on CPU.")

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()

    def embed_document_text(self, text: str, normalize: bool = True) -> List[List[float]]:
        # Simple chunking strategy
        sentences = text.split('. ')
        
        # In a more advanced scenario, you'd use a text splitter like RecursiveCharacterTextSplitter
        # For this, we'll just batch encode sentences.
        embeddings = self.model.encode(sentences, normalize_embeddings=normalize, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]