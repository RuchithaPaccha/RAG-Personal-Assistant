"""Document chunking module."""
from typing import List, Dict, Any
from src.config import ChunkingConfig


class DocumentChunker:
    """Chunks documents into smaller segments."""

    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk_text = text[i : i + chunk_size]
            if chunk_text.strip():
                chunk_data = {
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": min(i + chunk_size, len(text)),
                }
                if metadata:
                    chunk_data.update(metadata)
                chunks.append(chunk_data)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of ingested documents
            
        Returns:
            List of chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            doc_metadata = {k: v for k, v in doc.items() if k != "content"}
            chunks = self.chunk_text(doc["content"], metadata=doc_metadata)
            all_chunks.extend(chunks)

        return all_chunks
