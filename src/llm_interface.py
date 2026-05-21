"""LLM interface for generation."""
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
from src.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for open-source language models."""

    def __init__(self, config: LLMConfig = None):
        """
        Initialize LLM interface.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Remove prompt from output

    def build_rag_prompt(
        self,
        query: str,
        context: List[str],
        instruction: str = "Answer the question based on the provided context."
    ) -> str:
        """
        Build a RAG prompt with context.
        
        Args:
            query: User query
            context: Retrieved context chunks
            instruction: System instruction
            
        Returns:
            Formatted prompt
        """
        context_text = "\n".join([f"- {chunk}" for chunk in context])
        prompt = f"""System: {instruction}

Context:
{context_text}

Question: {query}

Answer:"""
        return prompt
