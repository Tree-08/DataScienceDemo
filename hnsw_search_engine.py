import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from my_algo.hnsw_retriever import HNSWRetriever
from retrieval_engine import SemanticSearchEngine


@dataclass
class HNSWConfig:
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    seed: int = 42


class CustomHNSWSearchEngine:
    """Adapter around the custom HNSW retriever with app-friendly outputs."""

    def __init__(self, base_engine: SemanticSearchEngine, config: Optional[HNSWConfig] = None) -> None:
        self.base_engine = base_engine
        self.config = config or HNSWConfig()
        self.retriever: Optional[HNSWRetriever] = None

    @classmethod
    def from_base_engine(
        cls,
        base_engine: SemanticSearchEngine,
        config: Optional[HNSWConfig] = None,
    ) -> "CustomHNSWSearchEngine":
        engine = cls(base_engine, config=config)
        engine.build_index()
        return engine

    def build_index(self) -> None:
        if self.base_engine.doc_embeddings is None:
            raise RuntimeError("Base engine embeddings are missing. Call load_or_build() first.")

        dim = int(self.base_engine.doc_embeddings.shape[1])
        retriever = HNSWRetriever(
            dim=dim,
            M=self.config.M,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
            seed=self.config.seed,
        )

        documents = [
            {
                "doc_id": i,
                "text": text,
            }
            for i, text in enumerate(self.base_engine.documents)
        ]
        retriever.add_documents(documents, self.base_engine.doc_embeddings)
        self.retriever = retriever

    def hnsw_search(self, query: str, k: int = 5, ef: Optional[int] = None) -> List[Dict[str, object]]:
        if k <= 0:
            return []
        if self.retriever is None:
            raise RuntimeError("HNSW index is not built.")

        query_vec = self.base_engine._encode_query(query)[0]
        hits = self.retriever.query(query_vec, k=min(k, len(self.base_engine.documents)), ef=ef)

        out: List[Dict[str, object]] = []
        for rank, (doc_obj, distance) in enumerate(hits, start=1):
            out.append(
                {
                    "rank": rank,
                    "doc_id": int(doc_obj["doc_id"]),
                    "score": -float(distance),
                    "distance": float(distance),
                    "text": str(doc_obj["text"]),
                }
            )
        return out

    def compare_with_faiss(self, query: str, k: int = 10, ef: Optional[int] = None) -> Dict[str, object]:
        start = time.perf_counter()
        faiss_hits = self.base_engine.semantic_search(query, k=k, use_exact=False)
        faiss_time_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        hnsw_hits = self.hnsw_search(query, k=k, ef=ef)
        hnsw_time_ms = (time.perf_counter() - start) * 1000.0

        faiss_ids = [hit["doc_id"] for hit in faiss_hits]
        hnsw_ids = [hit["doc_id"] for hit in hnsw_hits]

        overlap = len(set(faiss_ids).intersection(hnsw_ids))
        overlap_at_k = overlap / float(max(1, min(k, len(self.base_engine.documents))))

        return {
            "faiss": faiss_hits,
            "hnsw": hnsw_hits,
            "faiss_time_ms": float(faiss_time_ms),
            "hnsw_time_ms": float(hnsw_time_ms),
            "overlap_at_k": float(overlap_at_k),
        }
