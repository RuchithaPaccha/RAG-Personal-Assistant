# Streamlit app launcher script
#!/usr/bin/env python
"""Launch the Streamlit RAG application."""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting RAG Assistant Streamlit App...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
