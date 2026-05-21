"""Advanced retrieval module with improved search capabilities."""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
import json
from pathlib import Path
import logging
from src.config import RetrievalConfig
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """Advanced FAISS-based vector retrieval with enhancements."""

    def __init__(self, dimension: int, config: RetrievalConfig = None):
        """
        Initialize advanced retriever.
        
        Args:
            dimension: Embedding dimension
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.embeddings = None

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to index.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: Metadata for each embedding
        """
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        
        # Store embeddings for reranking
        if self.embeddings is None:
            self.embeddings = embeddings.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.metadata.extend(metadata)
        logger.info(f"Added {len(embeddings)} embeddings to index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        diversity: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar embeddings with optional diversity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            diversity: Whether to diversify results
            
        Returns:
            Tuple of (results metadata, distances)
        """
        top_k = top_k or self.config.top_k
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Initial search - get more results for filtering
        search_k = min(top_k * 3, len(self.metadata))
        distances, indices = self.index.search(query_embedding, search_k)
        distances = distances[0]
        indices = indices[0]

        # Filter by similarity threshold
        results = []
        for idx, distance in zip(indices, distances):
            # Convert L2 distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + distance)
            if similarity >= self.config.similarity_threshold:
                results.append((idx, similarity, self.metadata[idx]))

        # Sort by similarity
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Apply diversity if requested
        if diversity and len(results) > 1:
            results = self._apply_diversity(results, top_k)

        # Return top_k results
        final_results = [r[2] for r in results[:top_k]]
        final_distances = [r[1] for r in results[:top_k]]

        return final_results, final_distances

    def _apply_diversity(
        self,
        results: List[Tuple[int, float, Dict]],
        top_k: int,
        diversity_weight: float = 0.3
    ) -> List[Tuple[int, float, Dict]]:
        """
        Apply diversity to results to avoid redundant chunks.
        
        Args:
            results: List of (idx, similarity, metadata) tuples
            top_k: Number of top results to return
            diversity_weight: Weight for diversity vs relevance
            
        Returns:
            Diversified results
        """
        if not results:
            return results

        selected = [results[0]]
        results_copy = results[1:]

        while len(selected) < top_k and results_copy:
            best_idx = 0
            best_score = -1

            for i, (idx, sim, meta) in enumerate(results_copy):
                # Calculate diversity score (min distance to selected items)
                diversity_score = float('inf')
                if self.embeddings is not None and len(selected) > 0:
                    selected_embedding = self.embeddings[selected[0][0]].reshape(1, -1)
                    current_embedding = self.embeddings[idx].reshape(1, -1)
                    sim_to_selected = cosine_similarity(
                        selected_embedding, current_embedding
                    )[0][0]
                    diversity_score = 1.0 - sim_to_selected

                # Combined score: relevance + diversity
                combined_score = (1 - diversity_weight) * sim + diversity_weight * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i

            selected.append(results_copy[best_idx])
            results_copy.pop(best_idx)

        return selected

    def rerank(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using query-document similarity.
        
        Args:
            query_embedding: Query embedding
            candidates: Candidate documents
            
        Returns:
            Reranked candidates
        """
        if not candidates:
            return candidates

        # Calculate relevance scores
        scores = []
        for candidate in candidates:
            # Find the corresponding embedding in the index
            for i, meta in enumerate(self.metadata):
                if meta.get("text") == candidate.get("text"):
                    candidate_embedding = self.embeddings[i].reshape(1, -1)
                    query_emb = query_embedding.reshape(1, -1)
                    score = cosine_similarity(query_emb, candidate_embedding)[0][0]
                    scores.append(score)
                    break
            else:
                scores.append(0.0)

        # Sort by scores
        reranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        return reranked

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
    def load_index(cls, index_path: str, metadata_path: str) -> "AdvancedRetriever":
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            
        Returns:
            AdvancedRetriever instance
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
