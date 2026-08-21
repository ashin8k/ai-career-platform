from typing import Dict, List, Any


class LearningRoadmapGenerator:
    """
    Personalized Learning Roadmap Engine.
    Transforms detected missing skills into a structured week-by-week learning plan
    complete with topics, practice exercises, and interview target concepts.
    """

    SKILL_CURRICULUM_DATABASE = {
        "fastapi": {
            "title": "FastAPI & RESTful Microservices",
            "topics": ["Pydantic Data Validation", "Dependency Injection", "Async Endpoints", "API Documentation (Swagger/OpenAPI)"],
            "practice_problems": ["Build a CRUD REST API for resume metadata", "Implement OAuth2 JWT authentication"],
            "mini_project": "Build an asynchronous ML inference REST API using FastAPI."
        },
        "docker": {
            "title": "Docker & Containerization",
            "topics": ["Dockerfiles & Multi-stage builds", "Docker Compose for multi-container apps", "Volume mounts & Environment variables"],
            "practice_problems": ["Containerize a FastAPI app with PostgreSQL", "Optimize Docker image size below 200MB"],
            "mini_project": "Deploy a containerized Python ML microservice stack with Docker Compose."
        },
        "aws": {
            "title": "AWS Cloud Foundations for Engineers",
            "topics": ["AWS EC2 instance provisioning", "S3 object storage for model artifacts", "IAM policies & security groups"],
            "practice_problems": ["Upload model checkpoints to AWS S3 via boto3", "Deploy Docker container to AWS EC2"],
            "mini_project": "Host a containerized ML backend on AWS EC2 with S3 dataset storage."
        },
        "postgresql": {
            "title": "PostgreSQL & Database Design",
            "topics": ["Relational schemas & foreign keys", "SQL joins, indexes, and query optimization", "SQLAlchemy ORM integration"],
            "practice_problems": ["Design a schema for tracking resume match history", "Write raw SQL aggregations over user sessions"],
            "mini_project": "Integrate PostgreSQL database storage into a FastAPI backend using SQLAlchemy."
        },
        "kubernetes": {
            "title": "Kubernetes & Cloud Orchestration",
            "topics": ["Pods, Deployments, and Services", "kubectl CLI commands", "ConfigMaps & Secrets management"],
            "practice_problems": ["Deploy a 2-replica web service manifest", "Expose deployment via ClusterIP service"],
            "mini_project": "Deploy a auto-scaling microservice deployment on local Minikube."
        }
    }

    def generate_roadmap(self, missing_skills: List[str]) -> List[Dict[str, Any]]:
        """
        Generates a week-by-week roadmap ordered by missing skills priority.
        """
        roadmap: List[Dict[str, Any]] = []

        if not missing_skills:
            return [{
                "week": 1,
                "title": "Advanced Profile Optimization",
                "topics": ["System Architecture Design", "Performance Benchmarking", "Open Source Contributions"],
                "practice_problems": ["Optimize model inference latency", "Write unit test suites with >90% coverage"],
                "mini_project": "Contribute a feature or bugfix to an open-source ML repository."
            }]

        for idx, skill in enumerate(missing_skills, start=1):
            skill_clean = skill.lower()
            curriculum = self.SKILL_CURRICULUM_DATABASE.get(
                skill_clean,
                {
                    "title": f"Deep-Dive Learning: {skill.capitalize()}",
                    "topics": [f"Core principles of {skill}", f"Integration best practices for {skill}", "Debugging & Performance"],
                    "practice_problems": [f"Implement a hands-on module utilizing {skill}", f"Build a mini-demo testing {skill}"],
                    "mini_project": f"Build a functional portfolio application using {skill}."
                }
            )

            roadmap.append({
                "week": idx,
                "skill": skill,
                "title": curriculum["title"],
                "topics": curriculum["topics"],
                "practice_problems": curriculum["practice_problems"],
                "mini_project": curriculum["mini_project"]
            })

        return roadmap
