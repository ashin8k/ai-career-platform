from typing import Dict, List, Set


# Comprehensive Technical Skill Taxonomy with canonical naming and aliases
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "c++", "c#", "java", "javascript", "typescript", "sql", "r", "go",
        "rust", "html", "css", "bash", "shell", "kotlin", "swift", "scala", "matlab"
    ],
    "frameworks_libraries": [
        "fastapi", "flask", "django", "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "pandas", "numpy", "keras", "react", "react.js", "angular", "vue", "vue.js",
        "node.js", "express", "spring boot", "matplotlib", "seaborn", "opencv", "nltk",
        "spacy", "transformers", "sentence-transformers", "huggingface", "pydantic",
        "sqlalchemy", "optuna", "xgboost", "lightgbm", "catboost"
    ],
    "databases": [
        "postgresql", "postgres", "sqlite", "mysql", "mongodb", "redis", "cassandra",
        "elasticsearch", "faiss", "chromadb", "pinecone", "qdrant", "vector database"
    ],
    "cloud_devops": [
        "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "google cloud platform",
        "terraform", "git", "github", "gitlab", "ci/cd", "jenkins", "linux", "ec2",
        "s3", "lambda", "nginx", "bash"
    ],
    "ml_ai_concepts": [
        "machine learning", "deep learning", "natural language processing", "nlp",
        "computer vision", "reinforcement learning", "feature engineering",
        "model evaluation", "vector search", "rag", "llm", "prompt engineering",
        "neural networks", "transfer learning", "fine-tuning", "hyperparameter optimization",
        "classification", "regression", "clustering"
    ],
    "tools_methods": [
        "jira", "postman", "rest api", "restful api", "microservices",
        "object-oriented programming", "oop", "data structures", "algorithms",
        "system design", "agile", "scrum"
    ]
}

# Alias resolution mapping non-standard terms to canonical terms
SKILL_ALIASES: Dict[str, str] = {
    "postgres": "postgresql",
    "sklearn": "scikit-learn",
    "react.js": "react",
    "vue.js": "vue",
    "k8s": "kubernetes",
    "google cloud platform": "gcp",
    "restful api": "rest api",
    "nlp": "natural language processing"
}


def get_all_skills_flat() -> Set[str]:
    """Returns a set of all normalized skill strings in the taxonomy."""
    flat_skills = set()
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            flat_skills.add(skill.lower())
    return flat_skills


def normalize_skill_name(skill: str) -> str:
    """Maps skill variants to canonical skill names."""
    skill_clean = skill.strip().lower()
    return SKILL_ALIASES.get(skill_clean, skill_clean)
