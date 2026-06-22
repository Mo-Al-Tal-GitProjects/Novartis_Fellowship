"""Embedding generation and content-addressed artifact storage."""

from cipher.embeddings.generate import generate_embeddings
from cipher.embeddings.set import assemble_embedding_set

__all__ = ["assemble_embedding_set", "generate_embeddings"]
