"""Evaluation framework for RAG system."""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    factual_consistency: float
    answer_relevance: float


class Evaluator:
    """Evaluate RAG system performance."""

    @staticmethod
    def evaluate_retrieval(
        retrieved_chunks: List[str],
        relevant_chunks: List[str]
    ) -> Tuple[float, float, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_chunks: Chunks retrieved by system
            relevant_chunks: Ground truth relevant chunks
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)

        if not retrieved_set or not relevant_set:
            return 0.0, 0.0, 0.0

        true_positives = len(retrieved_set & relevant_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def evaluate_factual_consistency(
        generated_answer: str,
        reference_answer: str,
        embedding_model = None
    ) -> float:
        """
        Evaluate factual consistency using semantic similarity.
        
        Args:
            generated_answer: Generated answer from system
            reference_answer: Reference/gold answer
            embedding_model: Embedding model for similarity
            
        Returns:
            Consistency score (0-1)
        """
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        gen_embedding = embedding_model.encode([generated_answer])
        ref_embedding = embedding_model.encode([reference_answer])

        similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
        return float(max(0, similarity))

    @staticmethod
    def evaluate_answer_relevance(
        answer: str,
        query: str,
        embedding_model = None
    ) -> float:
        """
        Evaluate relevance of answer to query.
        
        Args:
            answer: Generated answer
            query: User query
            embedding_model: Embedding model
            
        Returns:
            Relevance score (0-1)
        """
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        query_embedding = embedding_model.encode([query])
        answer_embedding = embedding_model.encode([answer])

        similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
        return float(max(0, similarity))


class GoldenDatasetEvaluator:
    """Evaluate against golden dataset."""

    def __init__(self, golden_dataset_path: str):
        """
        Initialize evaluator with golden dataset.
        
        Args:
            golden_dataset_path: Path to golden dataset JSON
        """
        with open(golden_dataset_path, "r") as f:
            self.golden_data = json.load(f)

    def evaluate_sample(
        self,
        query_index: int,
        retrieved_chunks: List[str],
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample against golden data.
        
        Args:
            query_index: Index of query in golden dataset
            retrieved_chunks: Retrieved chunks
            generated_answer: Generated answer
            
        Returns:
            Evaluation results
        """
        sample = self.golden_data[query_index]

        # Retrieval evaluation
        precision, recall, f1 = Evaluator.evaluate_retrieval(
            retrieved_chunks, sample.get("relevant_chunks", [])
        )

        # Factual consistency
        consistency = Evaluator.evaluate_factual_consistency(
            generated_answer, sample["expected_answer"]
        )

        # Answer relevance
        relevance = Evaluator.evaluate_answer_relevance(
            generated_answer, sample["query"]
        )

        return {
            "query_index": query_index,
            "query": sample["query"],
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "retrieval_f1": f1,
            "factual_consistency": consistency,
            "answer_relevance": relevance,
            "overall_score": (precision + recall + consistency + relevance) / 4.0
        }

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate batch of results against golden data.
        
        Args:
            results: List of (query_index, retrieved_chunks, generated_answer) tuples
            
        Returns:
            Aggregated metrics
        """
        evaluations = []
        for result in results:
            eval_result = self.evaluate_sample(
                result["query_index"],
                result["retrieved_chunks"],
                result["generated_answer"]
            )
            evaluations.append(eval_result)

        # Aggregate metrics
        metrics = {
            "avg_retrieval_precision": np.mean([e["retrieval_precision"] for e in evaluations]),
            "avg_retrieval_recall": np.mean([e["retrieval_recall"] for e in evaluations]),
            "avg_retrieval_f1": np.mean([e["retrieval_f1"] for e in evaluations]),
            "avg_factual_consistency": np.mean([e["factual_consistency"] for e in evaluations]),
            "avg_answer_relevance": np.mean([e["answer_relevance"] for e in evaluations]),
            "avg_overall_score": np.mean([e["overall_score"] for e in evaluations]),
            "num_samples": len(evaluations)
        }

        return metrics
