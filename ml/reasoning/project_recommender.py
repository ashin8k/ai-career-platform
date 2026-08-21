from typing import Dict, List, Any


class ProjectRecommender:
    """
    Project Recommendation Engine.
    Analyzes missing skills and target JD requirements to synthesize high-impact portfolio projects.
    """

    PROJECT_TEMPLATES = [
        {
            "id": "ml_fastapi_docker",
            "title": "High-Throughput ML Prediction API with FastAPI & Docker",
            "required_gaps": ["fastapi", "docker"],
            "tech_stack": ["Python", "FastAPI", "Docker", "Scikit-learn", "Pytest"],
            "description": "Build an asynchronous microservice that exposes trained ML models over REST endpoints with Pydantic validation, containerized with multi-stage Dockerfiles.",
            "impact_rationale": "Demonstrates backend engineering expertise and containerization skills required for production ML deployments."
        },
        {
            "id": "postgres_sqlalchemy_api",
            "title": "Scalable Resume Analytics Platform with PostgreSQL & FastAPI",
            "required_gaps": ["postgresql", "fastapi"],
            "tech_stack": ["Python", "FastAPI", "PostgreSQL", "SQLAlchemy", "Alembic"],
            "description": "Design a relational schema to store user resumes, extracted skills, and candidate match scores with index optimization and ORM database migration.",
            "impact_rationale": "Proves ability to design production relational databases and integrate ORM persistence layers in web applications."
        },
        {
            "id": "aws_docker_deployment",
            "title": "Cloud-Native ML Microservice Infrastructure on AWS",
            "required_gaps": ["aws", "docker"],
            "tech_stack": ["AWS EC2", "AWS S3", "Docker", "Python", "CI/CD"],
            "description": "Containerize a Machine Learning application, automate Docker image builds via GitHub Actions, and deploy to an AWS EC2 instance connected to S3 object storage.",
            "impact_rationale": "Demonstrates DevOps, Cloud Infrastructure, and CI/CD automation highly sought after by engineering teams."
        },
        {
            "id": "faiss_vector_search",
            "title": "Semantic Vector Search & Candidate Matcher Engine",
            "required_gaps": ["faiss", "sentence-transformers"],
            "tech_stack": ["Python", "Sentence Transformers", "FAISS", "Streamlit"],
            "description": "Build a dense vector indexing system that converts text documents into 384-d embeddings and performs sub-millisecond similarity search using FAISS.",
            "impact_rationale": "Showcases modern AI Vector DB and Neural NLP engineering capabilities."
        }
    ]

    def recommend_projects(self, missing_skills: List[str], matching_skills: List[str]) -> List[Dict[str, Any]]:
        """
        Returns top 3 project recommendations prioritizing those that cover missing skills.
        """
        missing_set = set(s.lower() for s in missing_skills)
        recommendations = []

        for template in self.PROJECT_TEMPLATES:
            gaps_addressed = [g for g in template["required_gaps"] if g in missing_set]
            score = len(gaps_addressed) * 2 + len(template["required_gaps"])

            recommendations.append({
                "title": template["title"],
                "tech_stack": template["tech_stack"],
                "gaps_addressed": gaps_addressed if gaps_addressed else template["required_gaps"],
                "description": template["description"],
                "impact_rationale": template["impact_rationale"],
                "relevance_score": score
            })

        # Sort recommendations by relevance score descending
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return recommendations[:3]
