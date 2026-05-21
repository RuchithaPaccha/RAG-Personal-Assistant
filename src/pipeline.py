"""Main RAG pipeline."""
from typing import List, Dict, Any, Optional
from src.ingestion import DocumentIngester
from src.chunking import DocumentChunker
from src.embedding import EmbeddingGenerator
from src.retrieval import FAISSRetriever
from src.llm_interface import LLMInterface
from src.config import RAGConfig
from src.utils import setup_logging, ensure_directory_exists
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline."""

    def __init__(self, config: RAGConfig = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG system configuration
        """
        self.config = config or RAGConfig()
        setup_logging()

        # Initialize components
        self.embedding_generator = EmbeddingGenerator(self.config.embedding)
        self.retriever = FAISSRetriever(
            dimension=self.config.embedding.dimension,
            config=self.config.retrieval
        )
        self.llm = LLMInterface(self.config.llm)

        logger.info("RAG Pipeline initialized")

    def index_documents(self, directory_path: str) -> None:
        """
        Index documents from directory.
        
        Args:
            directory_path: Path to documents
        """
        ensure_directory_exists(self.config.data_dir)

        logger.info(f"Ingesting documents from {directory_path}")
        ingester = DocumentIngester()
        documents = ingester.ingest_directory(directory_path)

        if not documents:
            logger.warning("No documents ingested")
            return

        logger.info(f"Ingested {len(documents)} documents")

        logger.info("Chunking documents")
        chunker = DocumentChunker(self.config.chunking)
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Generating embeddings")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.encode_texts(chunk_texts)

        logger.info("Adding to index")
        self.retriever.add_embeddings(embeddings, chunks)

        logger.info("Saving index and metadata")
        self.retriever.save_index(self.config.index_path, self.config.metadata_path)

    def load_index(self) -> None:
        """Load existing FAISS index."""
        self.retriever = FAISSRetriever.load_index(
            self.config.index_path,
            self.config.metadata_path
        )

    def query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and retrieved chunks
        """
        top_k = top_k or self.config.retrieval.top_k

        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)

        # Retrieve relevant chunks
        retrieved_chunks, distances = self.retriever.search(query_embedding, top_k=top_k)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No relevant information found in the knowledge base.",
                "retrieved_chunks": [],
                "distances": []
            }

        # Extract chunk texts
        chunk_texts = [chunk["text"] for chunk in retrieved_chunks]

        # Build RAG prompt
        prompt = self.llm.build_rag_prompt(query, chunk_texts)

        # Generate answer
        answer = self.llm.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunk_texts,
            "distances": distances,
            "metadata": retrieved_chunks
        }

    def summarize_document(self, document_path: str) -> str:
        """
        Summarize a document.
        
        Args:
            document_path: Path to document
            
        Returns:
            Summary text
        """
        ingester = DocumentIngester()

        if document_path.lower().endswith(".pdf"):
            doc = ingester.ingest_pdf(document_path)
        elif document_path.lower().endswith(".docx"):
            doc = ingester.ingest_docx(document_path)
        else:
            doc = ingester.ingest_txt(document_path)

        # Create summarization prompt
        prompt = f"""Please provide a concise summary of the following document:

{doc['content'][:2000]}...

Summary:"""

        summary = self.llm.generate(prompt, max_tokens=300)
        return summary
