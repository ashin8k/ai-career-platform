import streamlit as st
import requests
import json
import os

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Career Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism, and dark mode badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-match {
        background-color: #065F46;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-missing {
        background-color: #991B1B;
        color: #FCA5A5;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .badge-partial {
        background-color: #92400E;
        color: #FCD34D;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Title Header
st.markdown('<div class="main-header">AI Career Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your resume and target job description for hybrid AI matching, ATS audit, personalized roadmaps & interview prep.</div>', unsafe_allow_html=True)

# Sidebar - Mode Selection
st.sidebar.header("⚙️ Execution Mode")
mode = st.sidebar.radio(
    "Choose Backend Processing",
    ["Direct ML Pipeline (Cloud & Local)", "Local FastAPI REST API"],
    index=0
)

# Initialize Session State
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "jd_data" not in st.session_state:
    st.session_state.jd_data = None
if "match_results" not in st.session_state:
    st.session_state.match_results = None

# Input Section (Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

with col2:
    st.subheader("2. Target Job Description")
    jd_title = st.text_input("Target Job Title", value="Machine Learning Engineer")
    jd_text = st.text_area(
        "Paste Target Job Description",
        value="""We are seeking a Machine Learning Engineer with 2+ years of experience in Python, PyTorch, FastAPI, Docker, and PostgreSQL.
Required: Python, SQL, Machine Learning, FastAPI, Docker, PostgreSQL.
Preferred: AWS EC2/S3, CI/CD pipelines, Kubernetes.""",
        height=160
    )

# Trigger Button
if st.button("🚀 Analyze Resume & Match Job", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a PDF resume file first!")
    elif not jd_text.strip():
        st.error("Please paste a target job description!")
    else:
        with st.spinner("Analyzing resume PDF, extracting skills, running hybrid matchers & ATS audit..."):
            run_direct = (mode == "Direct ML Pipeline (Cloud & Local)")
            
            if not run_direct:
                try:
                    # Attempt REST API endpoints
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res_upload = requests.post(f"{API_BASE_URL}/resume/upload", files=files, timeout=5)
                    if res_upload.status_code == 200:
                        resume_id = res_upload.json()["resume_id"]
                        res_jd = requests.post(
                            f"{API_BASE_URL}/job/analyze",
                            json={"title": jd_title, "job_description_text": jd_text},
                            timeout=5
                        )
                        jd_id = res_jd.json()["jd_id"]
                        res_match = requests.post(
                            f"{API_BASE_URL}/match",
                            json={"resume_id": resume_id, "jd_id": jd_id},
                            timeout=10
                        )
                        analysis_id = res_match.json()["analysis_id"]
                        res_details = requests.get(f"{API_BASE_URL}/analysis/{analysis_id}", timeout=5)
                        st.session_state.match_results = res_details.json()
                    else:
                        run_direct = True
                except Exception:
                    # Automatic fallback if local REST API is unreachable
                    run_direct = True

            if run_direct:
                # Direct ML Pipeline Execution (Works 100% self-contained in Cloud & Streamlit Cloud)
                from ml.preprocessing.pdf_extractor import PDFExtractor
                from ml.skill_extraction.skill_extractor import SkillExtractor
                from ml.skill_extraction.gap_analyzer import SkillGapAnalyzer
                from ml.skill_extraction.ats_analyzer import ATSAnalyzer
                from ml.matching.hybrid_matcher import HybridMatcher
                from ml.reasoning.ai_explainer import AIExplainer

                pdf_ext = PDFExtractor()
                pdf_res = pdf_ext.extract_text_from_bytes(uploaded_file.getvalue())
                raw_resume = pdf_res["text"]

                hybrid = HybridMatcher()
                match_res = hybrid.match(raw_resume, jd_text)
                gap_res = SkillGapAnalyzer().analyze_gap(raw_resume, jd_text)
                ats_res = ATSAnalyzer().analyze_resume(raw_resume, jd_text)
                report = AIExplainer().generate_full_analysis(raw_resume, jd_text, match_res, ats_res)

                st.session_state.match_results = {
                    "overall_match_percentage": match_res["overall_match_percentage"],
                    "skill_match_percentage": match_res["skill_match_percentage"],
                    "tfidf_match_percentage": match_res["tfidf_match_percentage"],
                    "semantic_embedding_percentage": match_res["semantic_embedding_percentage"],
                    "ats_score": ats_res["ats_score"],
                    "matching_skills": gap_res["matching_skills"],
                    "missing_skills": gap_res["missing_skills"],
                    "partial_skills": gap_res["partial_skills"],
                    "full_report": report
                }

            st.success("Analysis Complete!")

# Display Results Dashboard Tabs
if st.session_state.match_results is not None:
    res = st.session_state.match_results
    full_report = res.get("full_report", {})

    st.markdown("---")
    
    # 7 Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview & Match",
        "🎯 Skill Gap Matrix",
        "🛡️ ATS Analyzer",
        "🗺️ Learning Roadmap",
        "💡 Recommended Projects",
        "❓ Interview Preparation",
        "🔬 Model Approach Comparison"
    ])

    with tab1:
        st.header("Overall Match Rationale")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Match Score", f"{res.get('overall_match_percentage', 0)}%")
        m_col2.metric("Skill Match Rate", f"{res.get('skill_match_percentage', 0)}%")
        m_col3.metric("Lexical TF-IDF", f"{res.get('tfidf_match_percentage', 0)}%")
        m_col4.metric("ATS Score", f"{res.get('ats_score', 0)}/100")

        st.progress(int(res.get("overall_match_percentage", 0)))
        
        summary_exp = full_report.get("summary_explanation", {})
        st.info(summary_exp.get("match_rationale", "Calculated based on hybrid score."))

    with tab2:
        st.header("Skill Gap Analysis")
        sk_col1, sk_col2, sk_col3 = st.columns(3)
        
        with sk_col1:
            st.subheader("✅ Matching Skills")
            for s in res.get("matching_skills", []):
                st.markdown(f'<span class="badge-match">{s}</span>', unsafe_allow_html=True)
                
        with sk_col2:
            st.subheader("❌ Missing Skills")
            for s in res.get("missing_skills", []):
                st.markdown(f'<span class="badge-missing">{s}</span>', unsafe_allow_html=True)
                
        with sk_col3:
            st.subheader("⚠️ Partial Skills (Missing in Projects)")
            for s in res.get("partial_skills", []):
                st.markdown(f'<span class="badge-partial">{s}</span>', unsafe_allow_html=True)

    with tab3:
        st.header("ATS Resume Audit")
        st.metric("ATS Readiness Score", f"{res.get('ats_score', 0)} / 100")
        
        improvements = full_report.get("ats_improvements", [])
        if improvements:
            st.warning("Actionable ATS Improvements Required:")
            for imp in improvements:
                st.markdown(f"- 📌 {imp}")
        else:
            st.success("Great job! No major ATS formatting or keyword errors detected.")

    with tab4:
        st.header("Personalized Learning Roadmap")
        roadmap = full_report.get("learning_roadmap", [])
        for item in roadmap:
            with st.expander(f"Week {item.get('week', 1)}: {item.get('title', 'Learning Module')}", expanded=True):
                st.markdown(f"**Topics Covered:** {', '.join(item.get('topics', []))}")
                st.markdown(f"**Practice Problems:** {', '.join(item.get('practice_problems', []))}")
                st.markdown(f"**Mini Project:** {item.get('mini_project', '')}")

    with tab5:
        st.header("Recommended Portfolio Projects")
        projects = full_report.get("recommended_projects", [])
        for proj in projects:
            st.subheader(f"📌 {proj.get('title')}")
            st.markdown(f"**Tech Stack:** {', '.join(proj.get('tech_stack', []))}")
            st.markdown(f"**Description:** {proj.get('description')}")
            st.info(f"**Why this improves your profile:** {proj.get('impact_rationale')}")
            st.markdown("---")

    with tab6:
        st.header("Customized Interview Preparation")
        interview_qs = full_report.get("interview_preparation", {})
        
        diff_tab1, diff_tab2, diff_tab3 = st.tabs(["🟢 Easy Questions", "🟡 Medium Questions", "🔴 Difficult Questions"])
        with diff_tab1:
            for q in interview_qs.get("easy", []):
                st.markdown(f"**[{q.get('skill', 'General').upper()}] ({q.get('category')})**")
                st.write(q.get("question"))
                st.markdown("---")
        with diff_tab2:
            for q in interview_qs.get("medium", []):
                st.markdown(f"**[{q.get('skill', 'General').upper()}] ({q.get('category')})**")
                st.write(q.get("question"))
                st.markdown("---")
        with diff_tab3:
            for q in interview_qs.get("difficult", []):
                st.markdown(f"**[{q.get('skill', 'General').upper()}] ({q.get('category')})**")
                st.write(q.get("question"))
                st.markdown("---")

    with tab7:
        st.header("Algorithmic Approach Comparison")
        st.markdown("""
        | Algorithm Paradigm | Score Contribution | Key Strength | Known Weakness |
        | :--- | :---: | :--- | :--- |
        | **TF-IDF + Cosine** | 25% | Exact keyword lexical weighting | Cannot detect synonyms |
        | **Sentence Transformers** | 35% | Deep 384-d contextual semantics | May overestimate generic phrasing |
        | **Skill Graph Matching** | 40% | Deterministic set intersection | Ignores narrative experience depth |
        """)
