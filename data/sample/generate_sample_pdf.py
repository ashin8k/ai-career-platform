import os
import fitz  # PyMuPDF


SAMPLE_RESUME_TEXT = """
Alex Mercer
Computer Science & AI Engineering Student
Email: alex.mercer@example.com | Phone: +1-555-0199 | Location: San Francisco, CA
LinkedIn: linkedin.com/in/alex-mercer-ai | GitHub: github.com/alexmercer-ai

SUMMARY
Passionate 3rd-year Computer Science student specializing in AI/ML, NLP, and backend systems. Experienced in building end-to-end Machine Learning pipelines using Python, PyTorch, Scikit-learn, and FastAPI. Proficient in database management with PostgreSQL and containerization using Docker.

EDUCATION
Bachelor of Science in Computer Science (AI & ML Specialization)
University of Technology — Expected Graduation: May 2027
GPA: 3.85 / 4.0

TECHNICAL SKILLS
• Programming Languages: Python, C++, SQL, JavaScript, HTML/CSS
• Frameworks & Libraries: FastAPI, PyTorch, Scikit-learn, Pandas, NumPy, NLTK, Transformers, Node.js
• ML & Data Science: NLP, Sentence Transformers, Feature Engineering, Model Evaluation, Vector Search (FAISS)
• Databases & Cloud: PostgreSQL, SQLite, Docker, AWS (S3, EC2), Git, CI/CD pipelines

PROJECTS
AI Resume Matcher & Skill Gap Analyzer (Python, FastAPI, Scikit-learn, Docker)
• Built an end-to-end NLP matching platform using TF-IDF and Sentence Transformer embeddings.
• Extracted skill entities with regex and rule-based NLP parser.
• Developed REST APIs with FastAPI and designed an interactive Streamlit dashboard.

Predictive Customer Churn Engine (PyTorch, Pandas, PostgreSQL)
• Trained a deep learning binary classification model achieving 91.2% F1-score on 50k customer records.
• Conducted feature selection and optimized hyperparameters using Optuna.

EXPERIENCE
Machine Learning Intern — DataScale Labs (June 2025 – August 2025)
• Cleaned and preprocessed unstructured text datasets (100k+ records) using NLTK and pandas.
• Fine-tuned Hugging Face transformer models for sentiment classification.
• Dockerized backend endpoints for cloud deployment on AWS EC2.

CERTIFICATIONS
• Deep Learning Specialization — Coursera / DeepLearning.AI
• AWS Certified Cloud Practitioner — Amazon Web Services
"""


def create_sample_pdf(output_path: str):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 standard
    
    rect = fitz.Rect(40, 40, 555, 800)
    page.insert_textbox(rect, SAMPLE_RESUME_TEXT, fontsize=10, fontname="helv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Successfully generated sample resume PDF at: {output_path}")


if __name__ == "__main__":
    target_path = os.path.join(os.path.dirname(__file__), "sample_resume.pdf")
    create_sample_pdf(target_path)
