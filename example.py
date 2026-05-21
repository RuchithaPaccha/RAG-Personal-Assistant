"""Example usage of the RAG pipeline."""
if __name__ == "__main__":
    from src.pipeline import RAGPipeline
    from src.config import RAGConfig
    import json

    # Load configuration
    with open("config/config.json", "r") as f:
        config_dict = json.load(f)
    
    config = RAGConfig(
        embedding=RAGConfig.EmbeddingConfig(**config_dict["embedding"]),
        chunking=RAGConfig.ChunkingConfig(**config_dict["chunking"]),
        retrieval=RAGConfig.RetrievalConfig(**config_dict["retrieval"]),
        llm=RAGConfig.LLMConfig(**config_dict["llm"]),
        data_dir=config_dict["data_dir"],
        index_path=config_dict["index_path"],
        metadata_path=config_dict["metadata_path"]
    )

    # Initialize RAG pipeline
    rag = RAGPipeline(config)

    # Index documents (uncomment when you have documents in ./data)
    # rag.index_documents("./data")
    # rag.load_index()  # Load existing index

    # Example queries (uncomment when index is available)
    # result = rag.query("What are the main findings?")
    # print(f"Query: {result['query']}")
    # print(f"Answer: {result['answer']}")
    # print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
