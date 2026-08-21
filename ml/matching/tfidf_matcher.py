import numpy as np
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ml.preprocessing.text_cleaner import TextCleaner


class TFIDFMatcher:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) + Cosine Similarity Matcher.
    Constructs a sparse n-gram term feature matrix to compute lexical overlap between
    a cleaned resume and a target job description.
    """

    def __init__(self, ngram_range: tuple = (1, 2), cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner if cleaner is not None else TextCleaner()
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            preprocessor=self.cleaner.clean_text,
            sublinear_tf=True
        )

    def calculate_similarity(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Calculates cosine similarity score (range [0.0, 1.0]) between resume and job description text.
        Returns similarity score and top shared n-gram terms.
        """
        if not resume_text or not job_description_text:
            return {"score": 0.0, "top_shared_terms": []}

        corpus = [resume_text, job_description_text]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # Calculate Cosine Similarity: dot product over normalized vectors
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(sim_matrix[0][0])

        # Extract top shared feature words
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        vec1 = tfidf_matrix[0].toarray().flatten()
        vec2 = tfidf_matrix[1].toarray().flatten()
        
        overlap_weight = vec1 * vec2
        top_indices = np.argsort(overlap_weight)[::-1]
        
        top_terms = []
        for idx in top_indices:
            if overlap_weight[idx] > 0:
                top_terms.append(str(feature_names[idx]))
            if len(top_terms) >= 10:
                break

        return {
            "score": round(score, 4),
            "percentage": round(score * 100, 2),
            "top_shared_terms": top_terms
        }
