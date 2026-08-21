from typing import Dict, Any, Optional
from ml.matching.tfidf_matcher import TFIDFMatcher
from ml.matching.embedding_matcher import EmbeddingMatcher
from ml.matching.skill_matcher import SkillMatcher
from ml.skill_extraction.section_extractor import SectionExtractor


class HybridMatcher:
    """
    Hybrid Ensemble Resume-Job Matcher Engine.
    Fuses three complementary algorithmic paradigms:
      1. TF-IDF + Cosine Similarity (Lexical & Keyword exact weights)
      2. Sentence Transformers (Contextual Semantic similarity)
      3. Skill Set Graph Intersection (Hard-skill coverage ratio)
    """

    def __init__(
        self,
        weight_skill: float = 0.40,
        weight_embed: float = 0.35,
        weight_tfidf: float = 0.25,
        lazy_load_embedding: bool = True
    ):
        self.w_skill = weight_skill
        self.w_embed = weight_embed
        self.w_tfidf = weight_tfidf
        
        self.tfidf_matcher = TFIDFMatcher()
        self.skill_matcher = SkillMatcher()
        self.section_extractor = SectionExtractor()
        
        self._embed_matcher: Optional[EmbeddingMatcher] = None
        self.lazy_load_embedding = lazy_load_embedding
        if not lazy_load_embedding:
            self._embed_matcher = EmbeddingMatcher()

    @property
    def embed_matcher(self) -> EmbeddingMatcher:
        if self._embed_matcher is None:
            self._embed_matcher = EmbeddingMatcher()
        return self._embed_matcher

    def match(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Runs full hybrid matching pipeline and returns detailed breakdown scores.
        """
        # Step 1: Lexical TF-IDF match
        tfidf_res = self.tfidf_matcher.calculate_similarity(resume_text, job_description_text)

        # Step 2: Dense Semantic Embedding match
        embed_res = self.embed_matcher.calculate_similarity(resume_text, job_description_text)

        # Step 3: Skill Set Overlap match
        skill_res = self.skill_matcher.match_skills(resume_text, job_description_text)

        # Step 4: Section-specific relevance scoring
        sections = self.section_extractor.extract_sections(resume_text)
        exp_text = sections.get("experience", "")
        proj_text = sections.get("projects", "")

        exp_relevance = (
            self.embed_matcher.calculate_similarity(exp_text, job_description_text)["score"]
            if exp_text else embed_res["score"] * 0.8
        )
        proj_relevance = (
            self.embed_matcher.calculate_similarity(proj_text, job_description_text)["score"]
            if proj_text else embed_res["score"] * 0.8
        )

        # Step 5: Weighted Hybrid Score Calculation
        overall_score = (
            (self.w_skill * skill_res["score"]) +
            (self.w_embed * embed_res["score"]) +
            (self.w_tfidf * tfidf_res["score"])
        )
        overall_score = max(0.0, min(1.0, overall_score))

        missing_skill_pct = (
            (skill_res["missing_count"] / skill_res["jd_skill_count"] * 100)
            if skill_res["jd_skill_count"] > 0 else 0.0
        )

        return {
            "overall_match_score": round(overall_score, 4),
            "overall_match_percentage": round(overall_score * 100, 2),
            "skill_match_percentage": skill_res["percentage"],
            "tfidf_match_percentage": tfidf_res["percentage"],
            "semantic_embedding_percentage": embed_res["percentage"],
            "experience_relevance_percentage": round(exp_relevance * 100, 2),
            "project_relevance_percentage": round(proj_relevance * 100, 2),
            "missing_skill_percentage": round(missing_skill_pct, 2),
            "matching_skills": skill_res["matching_skills"],
            "missing_skills": skill_res["missing_skills"],
            "extra_skills": skill_res["extra_skills"],
            "top_tfidf_terms": tfidf_res["top_shared_terms"],
            "approach_comparison": {
                "tf_idf": {
                    "score": tfidf_res["score"],
                    "strengths": "Exact keyword lexical frequency matching.",
                    "weaknesses": "Misses synonyms, insensitive to word order or semantic intent."
                },
                "sentence_transformers": {
                    "score": embed_res["score"],
                    "strengths": "Understands deep contextual semantic meaning & domain synonyms.",
                    "weaknesses": "May give high similarity for generic phrasing without checking specific skills."
                },
                "skill_graph_matching": {
                    "score": skill_res["score"],
                    "strengths": "Deterministic, exact set-intersection for required tech stack.",
                    "weaknesses": "Ignores candidate experience depth and project narrative."
                }
            }
        }
