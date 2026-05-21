"""Utility functions for the RAG system."""
from typing import List, Dict, Any
import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, creating if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_golden_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load golden evaluation dataset.
    
    Args:
        dataset_path: Path to golden dataset JSON
        
    Returns:
        List of evaluation samples
    """
    import json
    with open(dataset_path, "r") as f:
        return json.load(f)


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
