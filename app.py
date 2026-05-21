"""Streamlit web interface for RAG system."""
import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
from src.pipeline import RAGPipeline
from src.config import RAGConfig, EmbeddingConfig, ChunkingConfig, RetrievalConfig, LLMConfig


# Page configuration
st.set_page_config(
    page_title="RAG Assistant - Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
    }
    .source-chunk {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.85em;
        border-left: 3px solid #FF9800;
    }
    .stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline with caching."""
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
        return RAGPipeline(config)
    except Exception as e:
        st.error(f"Failed to load RAG pipeline: {e}")
        return None


def save_chat_history(chat_history, filename="chat_history.json"):
    """Save chat history to JSON file."""
    history_dir = Path("chat_histories")
    history_dir.mkdir(exist_ok=True)
    
    filepath = history_dir / filename
    with open(filepath, "w") as f:
        json.dump(chat_history, f, indent=2)


def load_chat_history(filename="chat_history.json"):
    """Load chat history from JSON file."""
    history_dir = Path("chat_histories")
    filepath = history_dir / filename
    
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def get_chat_filenames():
    """Get list of saved chat histories."""
    history_dir = Path("chat_histories")
    if history_dir.exists():
        return sorted([f.stem for f in history_dir.glob("*.json")])
    return []


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Assistant - Personal Knowledge Base</h1>
        <p>Upload documents, ask questions, and get answers grounded in your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = load_rag_pipeline()
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = False
    if "current_chat_file" not in st.session_state:
        st.session_state.current_chat_file = "chat_history"
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Document Upload Section
        st.markdown("#### 📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload multiple documents to build your knowledge base"
        )
        
        if uploaded_files:
            # Save uploaded files
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = data_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)
            
            st.success(f"✓ Saved {len(saved_files)} file(s)")
            
            # Index documents button
            if st.button("🔍 Index Documents", use_container_width=True):
                with st.spinner("Indexing documents... This may take a moment"):
                    try:
                        if st.session_state.rag_pipeline:
                            st.session_state.rag_pipeline.index_documents("./data")
                            st.session_state.documents_indexed = True
                            st.success("✓ Documents indexed successfully!")
                    except Exception as e:
                        st.error(f"Error indexing documents: {e}")
        
        # Check if index exists
        index_path = Path("faiss_index/index.faiss")
        if index_path.exists():
            st.success("✓ Knowledge base indexed")
            st.session_state.documents_indexed = True
        
        # Chat history management
        st.markdown("#### 💬 Chat History")
        
        chat_files = get_chat_filenames()
        if chat_files:
            selected_chat = st.selectbox(
                "Select chat history",
                chat_files,
                key="chat_selector"
            )
            if selected_chat != st.session_state.current_chat_file:
                st.session_state.current_chat_file = selected_chat
                st.session_state.chat_history = load_chat_history(f"{selected_chat}.json")
                st.rerun()
        
        # Create new chat
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_chat_file = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
        
        # Clear chat history
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared")
        
        # Statistics
        st.markdown("#### 📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages", len(st.session_state.chat_history))
        with col2:
            st.metric("Status", "Ready" if st.session_state.documents_indexed else "Not indexed")
        with col3:
            st.metric("Chat File", st.session_state.current_chat_file[:15])
    
    # Main content area
    if not st.session_state.documents_indexed:
        st.warning(
            "⚠️ No indexed documents found. Please upload documents and click 'Index Documents' to get started.",
            icon="⚠️"
        )
        return
    
    # Chat interface
    st.markdown("### 💬 Chat with Your Knowledge Base")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant message with sources
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander(f"📖 Sources ({len(message['sources'])} chunks)"):
                        for j, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-chunk">
                            <strong>Source {j}:</strong> {source[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know?",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_pipeline.query(user_input, top_k=5)
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("retrieved_chunks", [])
                })
                
                # Save chat history
                save_chat_history(
                    st.session_state.chat_history,
                    f"{st.session_state.current_chat_file}.json"
                )
                
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; margin-top: 2rem;">
        <p>RAG Assistant • Powered by SentenceTransformers + FAISS + Open-Source LLM</p>
        <p>Upload your documents, build your knowledge base, and ask questions!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
