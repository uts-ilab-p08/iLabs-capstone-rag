"""Embedding model wrapper.

The reason this file exists is asymmetric embedding. BGE models are trained so
that *queries* carry an instruction prefix ("Represent this sentence for
searching relevant passages: ") while *documents* do not. Embedding a query the
same way as a document does not raise an error — it just quietly returns worse
results. Keeping both paths in one class means we cannot get it wrong by
accident: documents go through `embed_documents`, queries through `embed_query`.

fastembed runs the model with ONNX Runtime, so there is no PyTorch dependency.
Weights (~130MB for bge-base) download once on first use and are then cached.
"""

from __future__ import annotations

from fastembed import TextEmbedding

from rag.config import EMBED_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing. No instruction prefix."""
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. fastembed applies the model's query prefix."""
        return next(iter(self.model.query_embed(text))).tolist()
