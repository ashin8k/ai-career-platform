from typing import Dict, List, Any


class InterviewGenerator:
    """
    Targeted Technical Interview Question Generator.
    Produces questions categorized by Difficulty (Easy, Medium, Difficult) and Topic
    (Coding, Machine Learning, System Design, Project Deep-Dive) grounded in candidate skills and missing gaps.
    """

    QUESTION_BANK = {
        "python": {
            "easy": "Explain the difference between mutable and immutable data types in Python with examples.",
            "medium": "How do Python generators and decorators work under the hood? What are their memory benefits?",
            "difficult": "Explain the Python Global Interpreter Lock (GIL) and strategies to achieve true parallelism for CPU-bound tasks."
        },
        "fastapi": {
            "easy": "What is Pydantic and how does FastAPI use it for request payload validation?",
            "medium": "How does dependency injection work in FastAPI using `Depends()`?",
            "difficult": "Explain how FastAPI handles async event loops (`async def` vs `def`) and threadpool execution."
        },
        "docker": {
            "easy": "What is the difference between a Docker Image and a Docker Container?",
            "medium": "How do multi-stage Docker builds reduce container size in Python applications?",
            "difficult": "Explain Docker networking modes (bridge, host, overlay) and container security best practices."
        },
        "machine learning": {
            "easy": "What is the difference between supervised and unsupervised learning?",
            "medium": "How do you detect and fix overfitting in machine learning models? Explain bias-variance tradeoff.",
            "difficult": "Explain precision-recall AUC curves vs ROC AUC curves for imbalanced datasets."
        },
        "postgresql": {
            "easy": "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
            "medium": "How do B-tree database indexes accelerate query execution? What are the write performance tradeoffs?",
            "difficult": "Explain database ACID properties and transaction isolation levels in PostgreSQL."
        },
        "aws": {
            "easy": "What is the primary difference between AWS EC2 and AWS S3?",
            "medium": "How do you securely configure IAM roles for an application running on AWS EC2 to access AWS S3 without hardcoding API keys?",
            "difficult": "Architect a serverless, scalable ML inference pipeline using AWS Lambda, API Gateway, and S3."
        }
    }

    def generate_interview_questions(
        self,
        resume_skills: List[str],
        missing_skills: List[str]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates structured interview questions grouped by difficulty level.
        """
        all_relevant = list(set([s.lower() for s in resume_skills + missing_skills]))
        easy_q, medium_q, difficult_q = [], [], []

        for skill in all_relevant:
            if skill in self.QUESTION_BANK:
                q_data = self.QUESTION_BANK[skill]
                easy_q.append({"skill": skill, "category": "Coding/Fundamentals", "question": q_data["easy"]})
                medium_q.append({"skill": skill, "category": "ML/Engineering", "question": q_data["medium"]})
                difficult_q.append({"skill": skill, "category": "System Design/Advanced", "question": q_data["difficult"]})

        # Add generic project deep-dive question if bank items are small
        if not easy_q:
            easy_q.append({"skill": "General", "category": "Coding", "question": "Write a function to find duplicate elements in an array in O(n) time."})
            medium_q.append({"skill": "General", "category": "Project", "question": "Walk me through the architecture of your primary portfolio project."})
            difficult_q.append({"skill": "General", "category": "System Design", "question": "How would you scale a web API to handle 10,000 requests per second?"})

        return {
            "easy": easy_q,
            "medium": medium_q,
            "difficult": difficult_q
        }
