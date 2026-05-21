"""Configuration management for RAG system."""
from typing import Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    dimension: int = 384


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.5


@dataclass
class LLMConfig:
    """Language model configuration."""
    model_name: str = "mistralai/Mistral-7B"
    max_tokens: int = 512
    temperature: float = 0.7
    device: str = "cpu"


@dataclass
class RAGConfig:
    """Overall RAG system configuration."""
    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    llm: LLMConfig = None
    data_dir: str = "./data"
    index_path: str = "./faiss_index"
    metadata_path: str = "./metadata.json"

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.llm is None:
            self.llm = LLMConfig()

    @classmethod
    def from_json(cls, config_path: str) -> "RAGConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, output_path: str) -> None:
        """Save configuration to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=lambda o: o.__dict__)
