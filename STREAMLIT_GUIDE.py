"""Complete guide for using the RAG Assistant Streamlit Application."""
import os

STARTUP_GUIDE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                  RAG ASSISTANT - STREAMLIT APP GUIDE                       ║
╚════════════════════════════════════════════════════════════════════════════╝

🚀 QUICK START
==============

1. INSTALL STREAMLIT (if not already installed):
   pip install streamlit streamlit-chat

2. RUN THE APP:
   streamlit run app.py

   OR use the launcher:
   python run_app.py

3. OPEN IN BROWSER:
   The app will automatically open at: http://localhost:8501


📚 FEATURES
===========

✓ Document Upload
  - Upload PDF, TXT, and DOCX files
  - Multiple file uploads at once
  - Files saved to data/ folder automatically

✓ Document Indexing
  - Click "Index Documents" to create FAISS vector index
  - Embeddings generated automatically
  - FAISS index saved for reuse

✓ Smart Chat Interface
  - Ask questions about your documents
  - Get answers grounded in uploaded data
  - View source chunks for transparency

✓ Chat History Management
  - Automatic chat saving
  - Load previous conversations
  - Create new chat sessions
  - Clear chat history

✓ Vector Search
  - FAISS-powered semantic search
  - Top-5 relevant chunks retrieved per query
  - Configurable similarity thresholds


🎯 WORKFLOW
===========

Step 1: Upload Documents
  └─ Right sidebar → "Upload documents"
  └─ Select multiple PDF/TXT/DOCX files
  └─ Click "Index Documents"

Step 2: Ask Questions
  └─ Main chat area: "Ask a question about your documents"
  └─ Type your query, press Send
  └─ RAG system retrieves relevant chunks and generates answer

Step 3: Review Sources
  └─ Click "Sources" below each assistant message
  └─ View the document chunks used to generate the answer
  └─ Understand answer provenance

Step 4: Manage Conversations
  └─ Left sidebar → Select previous chats
  └─ Create new chats for different topics
  └─ Chat history automatically saved


📊 SIDEBAR FEATURES
===================

Document Management:
  • Upload documents → browse and select files
  • Index Documents → build FAISS index
  • Shows indexing progress
  • Displays "Knowledge base indexed" status

Chat History:
  • Select chat → load previous conversations
  • New Chat → start fresh conversation
  • Clear Current Chat → delete messages
  • Chat file selector

Statistics:
  • Messages count in current chat
  • System status (Ready/Not indexed)
  • Current chat file name


⚙️ CONFIGURATION
================

Edit config/config.json to customize:

{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
    "device": "cpu"
  },
  "chunking": {
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "retrieval": {
    "top_k": 5,
    "similarity_threshold": 0.5
  },
  "llm": {
    "model_name": "distilgpt2"
  }
}

Tuning Tips:
  • Increase top_k for more retrieved chunks (slower, more context)
  • Increase similarity_threshold for stricter filtering
  • Adjust chunk_size based on document type
  • Change embedding model for better quality (at cost of speed)


💾 CHAT HISTORY
===============

Locations:
  • Directory: chat_histories/
  • Files: chat_history.json, chat_<timestamp>.json
  • Format: JSON with messages and sources

Structure:
  {
    "role": "user/assistant",
    "content": "message text",
    "sources": ["chunk 1", "chunk 2"]  // only for assistant
  }

What's Saved:
  ✓ All user queries
  ✓ All system responses
  ✓ Retrieved document chunks (for source citation)
  ✓ Timestamps via filename


🔧 TROUBLESHOOTING
==================

Issue: "No indexed documents found"
  → Upload documents and click "Index Documents"
  → Wait for indexing to complete

Issue: App crashes on upload
  → Check file format (PDF, TXT, DOCX only)
  → Verify file isn't corrupted
  → Check data/ folder has write permissions

Issue: Slow responses
  → Reduce chunk_size in config.json
  → Reduce top_k for fewer retrieved chunks
  → Use a fast CPU or GPU

Issue: Poor answer quality
  → Add more documents
  → Adjust chunk_size and overlap
  → Try different embedding models
  → Improve document clarity


📈 BEST PRACTICES
==================

Documents:
  • Use clean, well-formatted documents
  • Remove images (if using text extraction)
  • Split very large PDFs into chunks
  • Include metadata/summaries

Queries:
  • Be specific and clear
  • Use complete sentences
  • Ask one question at a time
  • Rephrase if answer is poor

Configuration:
  • Start with defaults
  • Tune one parameter at a time
  • Monitor response quality
  • Save working configurations


🚫 KNOWN LIMITATIONS
====================

  • Streaming responses not yet implemented
  • No concurrent uploads
  • FAISS index must be rebuilt if config changes
  • Chat history stored locally (no cloud sync)
  • Single user (no multi-user support)


📚 FILE STRUCTURE
=================

app.py                     ← Main Streamlit app
run_app.py                 ← App launcher
src/
  ├── pipeline.py          ← Core RAG pipeline
  ├── embedding.py         ← Embeddings generation
  ├── retrieval.py         ← FAISS vector search
  ├── advanced_retrieval.py ← Enhanced retrieval
  ├── llm_interface.py     ← LLM integration
  └── ...
data/                      ← Your documents (auto-created)
faiss_index/               ← Vector index (auto-created)
chat_histories/            ← Chat logs (auto-created)
config/config.json         ← Configuration


🎓 EXAMPLES
===========

Example 1: Research Paper Analysis
  1. Upload research papers (PDFs)
  2. Index documents
  3. Ask: "What are the main findings?"
  4. Ask: "What methodology was used?"
  5. View sources for citations

Example 2: Personal Knowledge Base
  1. Upload notes, papers, articles (TXT/DOCX)
  2. Index documents
  3. Ask: "Tell me about [topic]"
  4. Build conversation on specific findings

Example 3: Meeting Transcripts
  1. Upload meeting transcripts (TXT)
  2. Index documents
  3. Ask: "What decisions were made?"
  4. Ask: "Who was responsible for X?"
  5. Extract action items


❓ FAQ
======

Q: Can I use the system offline?
A: Yes, everything runs locally. No cloud connection needed.

Q: How much RAM do I need?
A: Minimum 4GB, 8GB+ recommended for larger documents.

Q: Can I export chat history?
A: Yes, chat histories are JSON files in chat_histories/

Q: How do I reset everything?
A: Delete faiss_index/ and chat_histories/ folders

Q: Can I use different LLMs?
A: Yes, edit config/config.json llm.model_name

Q: Will my documents be uploaded anywhere?
A: No, all data stays on your computer locally.


📞 SUPPORT
===========

For issues or questions:
  1. Check TROUBLESHOOTING section above
  2. Review config/config.json settings
  3. Check data/ folder has documents
  4. Verify FAISS index exists in faiss_index/


═════════════════════════════════════════════════════════════════════════════

🎉 You're ready to use the RAG Assistant!
   Upload your documents, index them, and start asking questions.

═════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(STARTUP_GUIDE)
