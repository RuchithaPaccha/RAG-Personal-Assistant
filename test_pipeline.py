"""Test script to index documents and run RAG pipeline queries."""
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline
from src.config import RAGConfig, EmbeddingConfig, ChunkingConfig, RetrievalConfig, LLMConfig


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    
    print("=" * 70)
    print("RAG PIPELINE TEST - DOCUMENT INDEXING & QUERY")
    print("=" * 70)
    
    # Load configuration
    print("\n[1/5] Loading configuration...")
    try:
        with open("config/config.json", "r") as f:
            config_dict = json.load(f)
        
        config = RAGConfig(
            embedding=EmbeddingConfig(**config_dict["embedding"]),
            chunking=ChunkingConfig(**config_dict["chunking"]),
            retrieval=RetrievalConfig(**config_dict["retrieval"]),
            llm=LLMConfig(**config_dict["llm"]),
            data_dir=config_dict["data_dir"],
            index_path=config_dict["index_path"],
            metadata_path=config_dict["metadata_path"]
        )
        print(f"✓ Config loaded successfully")
        print(f"  Embedding Model: {config_dict['embedding']['model_name']}")
        print(f"  LLM Model: {config_dict['llm']['model_name']}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Initialize pipeline
    print("\n[2/5] Initializing RAG pipeline...")
    try:
        rag = RAGPipeline(config)
        print("✓ RAG Pipeline initialized")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Index documents
    print("\n[3/5] Indexing documents from ./data...")
    try:
        data_path = Path("./data")
        
        # Check for documents
        supported_files = list(data_path.glob("*.txt")) + \
                         list(data_path.glob("*.pdf")) + \
                         list(data_path.glob("*.docx"))
        
        if not supported_files:
            print("✗ No documents found in ./data")
            print("  Place .txt, .pdf, or .docx files in the data/ folder")
            return
        
        print(f"  Found {len(supported_files)} document(s):")
        for f in supported_files:
            print(f"    - {f.name}")
        
        print("\n  Indexing in progress (this may take a minute)...")
        rag.index_documents("./data")
        print("✓ Documents indexed successfully")
        print(f"  Index saved to: {config.index_path}")
    except Exception as e:
        print(f"✗ Failed to index documents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    print("\n[4/5] Running test queries...")
    print("-" * 70)
    
    test_queries = [
        "What is machine learning?",
        "Explain vector embeddings",
        "How does FAISS work?",
        "What are neural networks?",
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\nQuery {i}: \"{query}\"")
            result = rag.query(query, top_k=3)
            
            # Show answer
            answer = result['answer']
            answer_preview = (answer[:150] + "...") if len(answer) > 150 else answer
            print(f"  Answer: {answer_preview}")
            
            # Show retrieved chunks info
            print(f"  Retrieved: {len(result['retrieved_chunks'])} chunk(s)")
            
            results.append({
                "query": query,
                "answer": result['answer'],
                "chunks_retrieved": len(result['retrieved_chunks'])
            })
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
    
    print("\n" + "-" * 70)
    print(f"✓ Processed {len(results)}/{len(test_queries)} queries successfully")
    
    # Summary statistics
    if results:
        avg_chunks = sum(r['chunks_retrieved'] for r in results) / len(results)
        print(f"\nSummary Statistics:")
        print(f"  Average chunks retrieved: {avg_chunks:.1f}")
        print(f"  Total queries: {len(results)}")
    
    print("\n" + "=" * 70)
    print("✓ RAG PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Try your own queries: rag.query('your question')")
    print("  2. Add more documents to ./data/ folder")
    print("  3. Modify config/config.json to tune parameters")
    print("  4. Create evaluation/golden_dataset.json for evaluation")
    
    return results


if __name__ == "__main__":
    try:
        test_rag_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
