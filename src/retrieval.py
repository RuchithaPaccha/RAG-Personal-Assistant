"""Vector retrieval module using FAISS."""
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
import json
from pathlib import Path
import logging
from src.config import RetrievalConfig

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based vector retrieval."""

    def __init__(self, dimension: int, config: RetrievalConfig = None):
        """
        Initialize FAISS retriever.
        
        Args:
            dimension: Embedding dimension
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to index.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: Metadata for each embedding
        """
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: np.ndarray, top_k: int = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            Tuple of (results metadata, distances)
        """
        top_k = top_k or self.config.top_k
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)
        distances = distances[0]
        indices = indices[0]

        results = []
        for idx, distance in zip(indices, distances):
            if distance <= self.config.similarity_threshold * 1000:  # Normalize for L2
                results.append(self.metadata[idx])

        return results, distances.tolist()

    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load_index(cls, index_path: str, metadata_path: str) -> "FAISSRetriever":
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            
        Returns:
            FAISSRetriever instance
        """
        index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        retriever = cls(dimension=index.d)
        retriever.index = index
        retriever.metadata = metadata

        logger.info(f"Loaded index from {index_path}")
        logger.info(f"Loaded metadata from {metadata_path}")

        return retriever
