from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.db import engine, Base
from backend.api.routes import router as api_router

# Auto-create SQLite / PostgreSQL ORM tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Career Intelligence Platform API",
    description="Production-grade AI Platform for Resume PDF Parsing, Skill Extraction, Hybrid Matching, ATS Audit, and Personalized Career Roadmaps.",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "platform": "AI Career Intelligence Platform",
        "status": "online",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
