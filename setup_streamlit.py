"""Setup script for RAG Assistant Streamlit application."""
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = ["streamlit", "torch", "transformers", "sentence_transformers", "faiss"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package if package != "faiss" else "faiss")
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
    
    return len(missing) == 0


def setup_directories():
    """Create necessary directories."""
    print("\nSetting up directories...")
    
    directories = [
        "data",
        "faiss_index",
        "chat_histories",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ {directory}/")


def download_models():
    """Download required models."""
    print("\nPre-downloading models (may take a few minutes)...")
    
    try:
        print("  Downloading embedding model...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("  ✓ Embedding model ready")
        
        print("  Downloading LLM model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained("distilgpt2")
        AutoModelForCausalLM.from_pretrained("distilgpt2")
        print("  ✓ LLM model ready")
    except Exception as e:
        print(f"  ⚠️  Could not pre-download models: {e}")
        print("  Models will be downloaded on first use")


def main():
    """Run setup."""
    print("=" * 70)
    print("RAG ASSISTANT - STREAMLIT SETUP")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Failed to install dependencies")
        return False
    
    # Setup directories
    setup_directories()
    
    # Download models
    download_models()
    
    print("\n" + "=" * 70)
    print("✓ Setup completed successfully!")
    print("=" * 70)
    print("\nTo start the app, run:")
    print("  streamlit run app.py")
    print("\nOr use the launcher:")
    print("  python run_app.py")
    print("\nThe app will open at http://localhost:8501")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
