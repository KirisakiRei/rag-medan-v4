"""Pydantic models for embedding service."""
from pydantic import BaseModel, Field
from typing import List, Optional


class EmbedRequest(BaseModel):
    """Request body for /embed endpoint."""
    texts: List[str] = Field(..., description="List of texts to encode")
    prefix: str = Field(default="query: ", description="Prefix for E5 model (e.g. 'query: ' or 'passage: ')")
    model_size: str = Field(default="small", description="Model size: 'small' (384) or 'large' (1024)")


class EmbedResponse(BaseModel):
    """Response body for /embed endpoint."""
    embeddings: List[List[float]]
    dimension: int
    model_size: str
    count: int
