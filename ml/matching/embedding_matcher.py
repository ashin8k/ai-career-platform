import numpy as np
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingMatcher:
    """
    Dense Semantic Embedding Matcher.
    Uses Sentence Transformers ('all-MiniLM-L6-v2') to convert resume and JD text into
    384-dimensional dense contextual vectors, and measures cosine distance via FAISS or Dot Product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding(self, text: str) -> np.ndarray:
        """Encodes text into a normalized 384-d dense vector."""
        if not text or not text.strip():
            return np.zeros((384,), dtype=np.float32)
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def calculate_similarity(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Computes dense semantic similarity score using normalized inner product (Cosine Similarity).
        """
        vec_resume = self.get_embedding(resume_text)
        vec_jd = self.get_embedding(job_description_text)

        # Dot product of L2-normalized vectors is exact Cosine Similarity
        similarity = float(np.dot(vec_resume, vec_jd))
        # Clip score between 0.0 and 1.0
        similarity = max(0.0, min(1.0, similarity))

        return {
            "score": round(similarity, 4),
            "percentage": round(similarity * 100, 2),
            "embedding_dim": int(vec_resume.shape[0]),
            "model_name": self.model_name
        }

    def search_top_k_candidates(self, query_jd: str, resume_texts: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Demonstrates FAISS vector database search across multiple candidate resumes.
        """
        if not resume_texts:
            return []

        embeddings = np.vstack([self.get_embedding(r) for r in resume_texts])
        dim = embeddings.shape[1]

        # FAISS IndexFlatIP (Inner Product over normalized vectors = Cosine Similarity)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        query_vec = self.get_embedding(query_jd).reshape(1, -1)
        scores, indices = index.search(query_vec, min(k, len(resume_texts)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                "rank": rank + 1,
                "resume_index": int(idx),
                "similarity_score": round(float(score), 4)
            })

        return results
