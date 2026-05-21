"""Embedding generation module."""
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EmbeddingConfig
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using SentenceTransformers."""

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize embedding generator.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        logger.info(f"Loaded embedding model: {self.config.model_name}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
