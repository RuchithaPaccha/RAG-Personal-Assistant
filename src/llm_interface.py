"""LLM interface for generation."""
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
from src.config import LLMConfig

logger = logging.getLogger(__name__)

# Lightweight models for CPU inference
LIGHTWEIGHT_MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "distilbert": "distilbert-base-uncased"
}


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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,  # Use float32 for CPU stability
                low_cpu_mem_usage=True
            )
            logger.info(f"Successfully loaded model: {self.config.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")
            logger.info("Falling back to distilgpt2...")
            self.config.model_name = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            logger.info("Fallback model loaded successfully")

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

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 150),  # Cap at 150 for CPU performance
                temperature=max(temperature, 0.1),
                do_sample=temperature > 0,
                top_p=0.9,
                top_k=40,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()  # Remove prompt from output

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
