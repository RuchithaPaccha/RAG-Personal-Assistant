
A lightweight RAG (Retrieval-Augmented Generation) system that ingests PDFs, text files, and documents, then provides intelligent Q&A and summarization grounded in your personal knowledge base.

## Features

- **Multi-format Ingestion**: Process PDFs, DOCX, and text files
- **Smart Chunking**: Configurable document chunking with overlap
- **Embeddings**: SentenceTransformers for semantic text representations
- **Vector Search**: FAISS for efficient similarity searching
- **Open-Source LLMs**: Integration with Hugging Face models (Mistral, Llama, etc.)
- **Context-Aware Generation**: RAG prompting for factual, grounded answers
- **Evaluation Framework**: Golden dataset evaluation with precision, recall, consistency metrics

## Architecture

```
┌─────────────────┐
│  Documents      │
│ (PDF, DOCX)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunking      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │◄─── SentenceTransformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
└─────────────────┘
         ▲
         │
    ┌────┴───┐
    │  Query  │
    └────┬───┘
         │
         ▼
┌─────────────────┐
│  Retrieval      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM + Context  │◄─── Open-Source LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grounded       │
│    Answer       │
└─────────────────┘
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── ingestion.py           # Document ingestion
│   ├── chunking.py            # Text chunking strategies
│   ├── embedding.py           # Embedding generation
│   ├── retrieval.py           # FAISS retrieval
│   ├── llm_interface.py       # LLM interaction
│   ├── pipeline.py            # Complete RAG pipeline
│   └── utils.py               # Utility functions
├── evaluation/
│   ├── evaluator.py           # Evaluation framework
│   └── golden_dataset.json    # Golden evaluation dataset
├── config/
│   └── config.json            # Configuration file
├── data/                      # Store documents here
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. **Clone or create the project**:
   ```bash
   cd Rag\ assistant
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download embedding and LLM models** (first run will auto-download):
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
   ```

## Quick Start

### 1. Index Your Documents

```python
from src.pipeline import RAGPipeline
from src.config import RAGConfig

# Initialize pipeline
config = RAGConfig()
rag = RAGPipeline(config)

# Index documents from a directory
rag.index_documents("./data")
```

### 2. Query Your Knowledge Base

```python
# Ask a question
result = rag.query("What are the key findings in my research?")
print(f"Answer: {result['answer']}")
print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
```

### 3. Summarize Documents

```python
# Summarize a single document
summary = rag.summarize_document("./data/paper.pdf")
print(f"Summary: {summary}")
```

## Web Interface (Streamlit)

### 🚀 Launch the Web App

The easiest way to use RAG Assistant is through the Streamlit web interface:

```bash
# Install Streamlit dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### ✨ Web Interface Features

**Document Management:**
- 📤 Upload PDF, TXT, and DOCX files
- 🔍 Automatic document indexing with progress tracking
- 🗂️ Multiple file support in one batch

**Chat Interface:**
- 💬 Natural language Q&A with your documents
- 📖 View source chunks for every answer
- 🔗 Track which parts of documents are being used

**Chat History:**
- 💾 Automatic chat saving and recovery
- 🔄 Switch between multiple conversations
- 🗑️ Clear individual chats or start fresh

**Advanced Features:**
- ⚡ Vector search with similarity thresholds
- 🎯 Configurable retrieval parameters
- 📊 Real-time statistics and system status

### Use the Web App

1. **Upload Documents** (Sidebar → Upload documents)
   - Click "Browse files" and select PDF/TXT/DOCX
   - Select multiple files at once
   - Click "Index Documents" to build the knowledge base

2. **Ask Questions** (Main chat area)
   - Type your question
   - Press Send to get an answer
   - Click "Sources" to see relevant document chunks

3. **Manage Chats** (Sidebar → Chat History)
   - Create new chats for different topics
   - Load previous conversations
   - Save important chats

### Configuration

Edit `config/config.json` to customize the web app:

```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32
  },
  "retrieval": {
    "top_k": 5,
    "similarity_threshold": 0.5
  }
}
```

**Recommended Settings:**
- **top_k**: 3-5 for balanced context and speed
- **similarity_threshold**: 0.5 for high precision
- **chunk_size**: 512 for general documents

## Configuration

Edit `config/config.json` to customize:

### Embedding Model
```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "batch_size": 32
  }
}
```

### Chunking Strategy
```json
{
  "chunking": {
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

### LLM Settings
```json
{
  "llm": {
    "model_name": "mistralai/Mistral-7B",
    "max_tokens": 512,
    "temperature": 0.7
  }
}
```

## Evaluation

### Create Golden Dataset

Edit `evaluation/golden_dataset.json` with your evaluation queries:

```json
[
  {
    "query": "What is the main topic?",
    "expected_answer": "The paper focuses on...",
    "relevant_chunks": ["chunk_text_1", "chunk_text_2"]
  }
]
```

### Run Evaluation

```python
from evaluation.evaluator import GoldenDatasetEvaluator

evaluator = GoldenDatasetEvaluator("evaluation/golden_dataset.json")

results = [
    {
        "query_index": 0,
        "retrieved_chunks": [...],
        "generated_answer": "..."
    }
]

metrics = evaluator.evaluate_batch(results)
print(f"Average F1 Score: {metrics['avg_retrieval_f1']}")
print(f"Factual Consistency: {metrics['avg_factual_consistency']}")
```

## Evaluation Metrics

- **Retrieval Precision**: % of retrieved chunks that are relevant
- **Retrieval Recall**: % of relevant chunks that were retrieved
- **Retrieval F1**: Harmonic mean of precision and recall
- **Factual Consistency**: Semantic similarity between generated and reference answers
- **Answer Relevance**: Semantic similarity between query and answer

## Performance Tips

1. **Use smaller embedding models** for CPU: `all-MiniLM-L6-v2` (384 dims) vs `all-mpnet-base-v2` (768 dims)
2. **Quantize models** for faster inference on resource-constrained systems
3. **Adjust chunk size**: Smaller chunks = better retrieval precision, larger chunks = better context
4. **Tune similarity threshold** in `config.json` to balance precision/recall
5. **Use GPU**: Set `device: "cuda"` if CUDA is available

## Example Use Cases

- Research paper Q&A system
- Meeting transcript analysis
- Personal note retrieval
- Documentation assistant
- Knowledge base search

## Future Enhancements

- [ ] Multi-modal embeddings (images + text)
- [ ] Query expansion and reformulation
- [ ] Hierarchical chunking strategies
- [ ] Re-ranking with cross-encoders
- [ ] Streaming response generation
- [ ] Fine-tuning on domain-specific data
- [ ] Web interface with Streamlit/Gradio
- [ ] Batch inference optimization

## Contributing

Improvements welcome! Areas for enhancement:
- Better chunking strategies
- Hybrid search (BM25 + semantic)
- Advanced prompting techniques
- Extended evaluation metrics

## License

MIT License - feel free to use for personal or commercial projects.

## Acknowledgments

- SentenceTransformers for embeddings
- FAISS for efficient similarity search
- Hugging Face for open-source models
- LangChain community for RAG patterns
