import json
import os
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ml.matching.tfidf_matcher import TFIDFMatcher
from ml.matching.embedding_matcher import EmbeddingMatcher
from ml.matching.skill_matcher import SkillMatcher
from ml.matching.hybrid_matcher import HybridMatcher


class MLEvaluator:
    """
    ML Evaluation Benchmark Suite.
    Evaluates Precision, Recall, F1, Accuracy, and Ground-Truth Correlation across
    TF-IDF, Sentence Transformers, Skill Graph, and Hybrid Ensemble algorithms.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "sample", "evaluation_dataset.json"
            )
        self.dataset_path = dataset_path

        self.tfidf_matcher = TFIDFMatcher()
        self.skill_matcher = SkillMatcher()
        self.hybrid_matcher = HybridMatcher(lazy_load_embedding=True)
        self._embed_matcher: Optional[EmbeddingMatcher] = None

    @property
    def embed_matcher(self) -> EmbeddingMatcher:
        if self._embed_matcher is None:
            self._embed_matcher = EmbeddingMatcher()
        return self._embed_matcher

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def evaluate_all(self, threshold: float = 0.50) -> Dict[str, Any]:
        dataset = self.load_dataset()
        y_true_binary = [item["ground_truth_relevance"] for item in dataset]
        y_true_continuous = [item["ground_truth_similarity"] for item in dataset]

        results = {}
        approaches = ["tf_idf", "sentence_transformers", "skill_graph", "hybrid_ensemble"]

        for approach in approaches:
            scores = []
            for item in dataset:
                r_text = item["resume_text"]
                jd_text = item["jd_text"]

                if approach == "tf_idf":
                    s = self.tfidf_matcher.calculate_similarity(r_text, jd_text)["score"]
                elif approach == "sentence_transformers":
                    s = self.embed_matcher.calculate_similarity(r_text, jd_text)["score"]
                elif approach == "skill_graph":
                    s = self.skill_matcher.match_skills(r_text, jd_text)["score"]
                else:  # hybrid_ensemble
                    s = self.hybrid_matcher.match(r_text, jd_text)["overall_match_score"]

                scores.append(s)

            y_pred_binary = [1 if s >= threshold else 0 for s in scores]

            prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            acc = float(accuracy_score(y_true_binary, y_pred_binary))

            # Pearson Correlation
            corr = float(np.corrcoef(scores, y_true_continuous)[0, 1]) if len(scores) > 1 else 0.0

            results[approach] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(acc, 4),
                "correlation": round(corr, 4),
                "avg_predicted_score": round(float(np.mean(scores)), 4)
            }

        return results
