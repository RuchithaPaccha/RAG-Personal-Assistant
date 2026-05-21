<!-- Use this file to provide workspace-specific custom instructions to Copilot. -->

## Personal Knowledge Assistant - RAG System Project

### Project Overview
A personal knowledge assistant that ingests PDFs and text files, chunks content, generates embeddings using SentenceTransformers, stores them in FAISS, and uses open-source LLMs with context-aware prompting for Q&A and summarization. Includes evaluation loop with golden dataset.

### Key Technologies
- **Embeddings**: SentenceTransformers
- **Vector Store**: FAISS
- **Document Processing**: PyPDF, python-docx, txt parsing
- **LLM**: Hugging Face transformers (Mistral, Llama, or similar)
- **RAG Framework**: LangChain or custom pipeline
- **Evaluation**: Custom evaluation loop with golden dataset

### Development Guidelines
- Use Python 3.9+
- Maintain modular architecture (ingestion, chunking, embedding, retrieval, generation)
- Include type hints in code
- Create comprehensive evaluation metrics
- Document all configuration options
