"""Document ingestion module for PDF and text files."""
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import docx
import logging

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingests PDFs, DOCX, and text files."""

    @staticmethod
    def ingest_pdf(file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dict with document content and metadata
        """
        content = []
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                content.append(text)

        return {
            "content": "\n".join(content),
            "source": str(file_path),
            "type": "pdf",
            "pages": len(pdf_reader.pages)
        }

    @staticmethod
    def ingest_docx(file_path: str) -> Dict[str, Any]:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Dict with document content and metadata
        """
        doc = docx.Document(file_path)
        content = [paragraph.text for paragraph in doc.paragraphs]

        return {
            "content": "\n".join(content),
            "source": str(file_path),
            "type": "docx",
            "paragraphs": len(content)
        }

    @staticmethod
    def ingest_txt(file_path: str) -> Dict[str, Any]:
        """
        Extract text from TXT file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dict with document content and metadata
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "source": str(file_path),
            "type": "txt",
            "lines": len(content.splitlines())
        }

    @classmethod
    def ingest_directory(cls, directory_path: str) -> List[Dict[str, Any]]:
        """
        Ingest all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of ingested documents
        """
        documents = []
        directory = Path(directory_path)

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == ".pdf":
                        doc = cls.ingest_pdf(str(file_path))
                    elif file_path.suffix.lower() == ".docx":
                        doc = cls.ingest_docx(str(file_path))
                    elif file_path.suffix.lower() == ".txt":
                        doc = cls.ingest_txt(str(file_path))
                    else:
                        continue
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error ingesting {file_path}: {e}")

        return documents
