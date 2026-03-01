"""
Token Estimator Module
======================

Accurate token counting for various LLM models using tiktoken.
Provides token estimation and budget enforcement for context prioritization.

Author: vex-memory team
Version: 1.0.0
"""

import tiktoken
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenEstimate:
    """Result of token estimation."""
    text: str
    token_count: int
    model: str
    truncated: bool = False
    original_length: Optional[int] = None


class TokenEstimator:
    """Accurate token counting for various LLM models."""
    
    # Map model names to tiktoken encodings
    SUPPORTED_MODELS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-2": "cl100k_base",  # Approximation
        "claude-3": "cl100k_base",  # Approximation
        "claude-3-opus": "cl100k_base",
        "claude-3-sonnet": "cl100k_base",
        "claude-3-haiku": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
    }
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize token estimator.
        
        Args:
            model: Model name for token counting (default: gpt-4)
        """
        self.model = model
        self.encoding_name = self.SUPPORTED_MODELS.get(model, "cl100k_base")
        
        try:
            self.encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {self.encoding_name}, using cl100k_base: {e}")
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using character approximation: {e}")
            # Fallback: ~4 characters per token
            return len(text) // 4
    
    def estimate_memory(self, memory: Dict[str, Any]) -> TokenEstimate:
        """Estimate tokens for a memory object.
        
        Args:
            memory: Memory dictionary with content, metadata, etc.
            
        Returns:
            TokenEstimate with count and formatted text
        """
        formatted = self._format_for_context(memory)
        count = self.count_tokens(formatted)
        
        return TokenEstimate(
            text=formatted,
            token_count=count,
            model=self.model,
            truncated=False
        )
    
    def truncate_to_budget(self, text: str, budget: int) -> TokenEstimate:
        """Truncate text to fit token budget.
        
        Args:
            text: Text to truncate
            budget: Maximum tokens allowed
            
        Returns:
            TokenEstimate with truncated text if needed
        """
        if not text or budget <= 0:
            return TokenEstimate("", 0, self.model, False)
        
        original_length = len(text)
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= budget:
            return TokenEstimate(
                text=text,
                token_count=len(tokens),
                model=self.model,
                truncated=False,
                original_length=original_length
            )
        
        # Reserve 3 tokens for ellipsis "..."
        truncate_at = max(0, budget - 3)
        truncated_tokens = tokens[:truncate_at]
        
        try:
            truncated_text = self.encoder.decode(truncated_tokens) + "..."
        except Exception as e:
            logger.warning(f"Decoding failed, using character truncation: {e}")
            # Fallback to character truncation
            char_budget = budget * 4
            truncated_text = text[:char_budget] + "..."
        
        return TokenEstimate(
            text=truncated_text,
            token_count=budget,
            model=self.model,
            truncated=True,
            original_length=original_length
        )
    
    def estimate_batch(self, memories: List[Dict[str, Any]]) -> List[TokenEstimate]:
        """Estimate tokens for multiple memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            List of TokenEstimate objects
        """
        return [self.estimate_memory(m) for m in memories]
    
    def _format_for_context(self, memory: Dict[str, Any]) -> str:
        """Format memory as it would appear in context.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Formatted string for context inclusion
        """
        parts = []
        
        # Add timestamp if available
        if memory.get("event_time"):
            event_time = memory["event_time"]
            # Handle both datetime objects and strings
            if hasattr(event_time, 'strftime'):
                timestamp_str = event_time.strftime('%Y-%m-%d')
            else:
                timestamp_str = str(event_time)[:10]  # Take first 10 chars (YYYY-MM-DD)
            parts.append(f"[{timestamp_str}]")
        
        # Add content
        content = memory.get("content", "")
        parts.append(content)
        
        # Add importance score if significant
        importance = memory.get("importance_score")
        if importance is not None and importance > 0.7:
            parts.append(f"(importance: {importance:.2f})")
        
        # Add metadata if present and small
        metadata = memory.get("metadata")
        if metadata and isinstance(metadata, dict):
            # Only include compact metadata
            if len(str(metadata)) < 100:
                parts.append(f"[metadata: {metadata}]")
        
        return " ".join(parts)
    
    def get_available_budget(
        self,
        total_budget: int,
        prefix: str = "",
        suffix: str = ""
    ) -> int:
        """Calculate available token budget after accounting for prefix/suffix.
        
        Args:
            total_budget: Total token budget
            prefix: Text to prepend to context
            suffix: Text to append to context
            
        Returns:
            Available tokens for memories
        """
        prefix_tokens = self.count_tokens(prefix) if prefix else 0
        suffix_tokens = self.count_tokens(suffix) if suffix else 0
        
        available = total_budget - prefix_tokens - suffix_tokens
        return max(0, available)


# Convenience function for quick token counting
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Quick token count without creating estimator instance.
    
    Args:
        text: Text to count
        model: Model name
        
    Returns:
        Token count
    """
    estimator = TokenEstimator(model)
    return estimator.count_tokens(text)
