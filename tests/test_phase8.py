import pytest
from ml.evaluation.evaluator import MLEvaluator


def test_ml_evaluator_benchmark():
    evaluator = MLEvaluator()
    results = evaluator.evaluate_all(threshold=0.50)

    assert "tf_idf" in results
    assert "sentence_transformers" in results
    assert "skill_graph" in results
    assert "hybrid_ensemble" in results

    # Hybrid ensemble should achieve high F1 and accuracy
    hybrid = results["hybrid_ensemble"]
    assert hybrid["accuracy"] >= 0.75
    assert hybrid["f1_score"] >= 0.75
