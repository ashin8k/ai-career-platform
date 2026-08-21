import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# PostgreSQL connection string default with SQLite fallback for offline local testing
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:////Users/ashink/.gemini/antigravity/scratch/ai-career-platform/ai_career_platform.db"
)

# Connect args needed for SQLite threading compatibility
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
