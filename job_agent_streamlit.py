import os
import io
import json
import re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from openai import OpenAI

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="AI Job Search Agent", page_icon="💼", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# -----------------------------
# Helpers
# -----------------------------
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed. Run: pip install PyPDF2")
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs).strip()


def extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported resume format. Please upload PDF, DOCX, or TXT.")


SCHEMA = {
    "type": "json_schema",
    "name": "job_search_plan",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_summary": {"type": "string"},
            "target_titles": {
                "type": "array",
                "items": {"type": "string"}
            },
            "skills_keywords": {
                "type": "array",
                "items": {"type": "string"}
            },
            "search_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source": {"type": "string"},
                        "query": {"type": "string"}
                    },
                    "required": ["source", "query"]
                }
            }
        },
        "required": ["candidate_summary", "target_titles", "skills_keywords", "search_queries"]
    }
}


def build_search_plan(
    resume_text: str,
    expected_roles: str,
    expected_designations: str,
    locations: str,
) -> Dict[str, Any]:
    client = get_openai_client()

    prompt = f"""
You are helping build a job-search agent.

Inputs:
1) Resume text
2) Expected roles
3) Expected designations
4) Preferred locations

Create:
- a concise candidate summary
- target job titles
- core skill keywords
- search queries for multiple sources: LinkedIn, Indeed, Glassdoor, company careers, and general web

Rules:
- Prefer seniority consistent with the resume and user intent
- Generate practical search strings
- Include location when available
- Keep queries optimized for public web search APIs
- Return valid JSON only

Expected roles:
{expected_roles}

Expected designations:
{expected_designations}

Preferred locations:
{locations}

Resume:
{resume_text[:18000]}
"""

    response = client.responses.create(
        timeout=60,
        model=DEFAULT_MODEL,
        input=prompt,
        text={"format": SCHEMA},
    )

    return json.loads(response.output_text)


# Fallback web search (no API key required) using DuckDuckGo HTML results
DDG_ENDPOINT = "https://duckduckgo.com/html/"
ADZUNA_ENDPOINT = "https://api.adzuna.com/v1/api/jobs/us/search/1"


def adzuna_search(query: str, location: str, num: int = 10) -> List[Dict[str, Any]]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": min(num, 20),
        "what": query,
        "where": location or "United States",
        "content-type": "application/json",
    }

    try:
        resp = requests.get(ADZUNA_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for item in data.get("results", []):
            jobs.append({
                "title": item.get("title", "Untitled Job"),
                "link": item.get("redirect_url") or item.get("adref", ""),
                "snippet": item.get("description", "")[:300],
                "displayed_link": item.get("company", {}).get("display_name", "Adzuna"),
            })
        return jobs
    except Exception:
        return []


def serp_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback DuckDuckGo search. This is best-effort only and may be unreliable on cloud hosts.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    params = {
        "q": query
    }

    try:
        resp = requests.post(DDG_ENDPOINT, data=params, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text

        results = []
        links = re.findall(r'<a[^>]+class="result__a"[^>]+href="(.*?)"[^>]*>(.*?)</a>', html)

        for link, title in links[:num]:
            clean_title = re.sub('<.*?>', '', title)
            results.append({
                "title": clean_title,
                "link": link,
                "snippet": "",
                "displayed_link": link
            })

        return results
    except Exception:
        return []


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = re.sub(r"#.*$", "", url)
    url = re.sub(r"\?.*$", "", url)
    return url.rstrip("/")


def collect_jobs(search_plan: Dict[str, Any], max_per_query: int = 8, location: str = "United States") -> List[Dict[str, Any]]:
    all_jobs: List[Dict[str, Any]] = []
    seen = set()

    for item in search_plan.get("search_queries", []):
        source = item.get("source", "web")
        query = item.get("query", "")
        if not query:
            continue

        results = []
        if ADZUNA_APP_ID and ADZUNA_APP_KEY:
            results = adzuna_search(query=query, location=location, num=max_per_query)

        if not results:
            results = serp_search(query, num=max_per_query)

        for r in results:
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            display_link = r.get("displayed_link", "")
            key = normalize_url(link)
            if not key or key in seen:
                continue
            seen.add(key)
            all_jobs.append(
                {
                    "source": source,
                    "title": title,
                    "url": link,
                    "snippet": snippet,
                    "display_link": display_link,
                }
            )

    return all_jobs


RANK_SCHEMA = {
    "type": "json_schema",
    "name": "ranked_jobs",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "fit_score": {"type": "integer"},
                        "reason": {"type": "string"},
                        "match_highlights": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["index", "fit_score", "reason", "match_highlights"]
                }
            }
        },
        "required": ["jobs"]
    }
}


def rank_jobs(
    resume_text: str,
    expected_roles: str,
    expected_designations: str,
    jobs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    client = get_openai_client()
    trimmed_jobs = jobs[:50]

    prompt = {
        "resume_text": resume_text[:14000],
        "expected_roles": expected_roles,
        "expected_designations": expected_designations,
        "jobs": [
            {
                "index": i,
                "title": j["title"],
                "url": j["url"],
                "snippet": j["snippet"],
                "source": j["source"],
            }
            for i, j in enumerate(trimmed_jobs)
        ],
        "task": "Score each job for relevance to the candidate from 0 to 100, based on likely match to resume and stated goals."
    }

    response = client.responses.create(
        timeout=60,
        model=DEFAULT_MODEL,
        input=json.dumps(prompt),
        text={"format": RANK_SCHEMA},
    )

    ranked = json.loads(response.output_text)["jobs"]
    merged = []
    for item in ranked:
        idx = item["index"]
        if 0 <= idx < len(trimmed_jobs):
            merged.append({**trimmed_jobs[idx], **item})

    merged.sort(key=lambda x: x.get("fit_score", 0), reverse=True)
    return merged


def generate_default_queries(plan: Dict[str, Any], locations: str) -> Dict[str, Any]:
    titles = plan.get("target_titles", [])[:4]
    if not titles:
        titles = ["QA Director", "Quality Engineering Director"]

    loc = locations.strip() if locations else "United States"
    queries = []

    for title in titles:
        queries.append({
            "source": "linkedin",
            "query": f'{title} jobs {loc} site:linkedin.com'
        })
        queries.append({
            "source": "indeed",
            "query": f'{title} jobs {loc} site:indeed.com'
        })
        queries.append({
            "source": "glassdoor",
            "query": f'{title} jobs {loc} site:glassdoor.com'
        })
        queries.append({
            "source": "general",
            "query": f'{title} jobs {loc}'
        })

    plan["search_queries"] = queries
    return plan

    titles = plan.get("target_titles", [])[:4]
    if not titles:
        titles = ["Software QA Director", "Quality Engineering Director"]

    loc = locations.strip() if locations else "United States"
    queries = []
    for title in titles:
        queries.append({
            "source": "linkedin",
            "query": f'site:linkedin.com/jobs/view "{title}" "{loc}"'
        })
        queries.append({
            "source": "indeed",
            "query": f'site:indeed.com/viewjob OR site:indeed.com/jobs "{title}" "{loc}"'
        })
        queries.append({
            "source": "glassdoor",
            "query": f'site:glassdoor.com/job-listing "{title}" "{loc}"'
        })
        queries.append({
            "source": "company-careers",
            "query": f'("careers" OR "jobs") "{title}" "{loc}"'
        })
    plan["search_queries"] = queries
    return plan


# -----------------------------
# UI
# -----------------------------
st.title("💼 AI Job Search Agent")
st.caption("Upload a resume, describe target roles, and get ranked job links from the web.")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    max_results = st.slider("Max jobs to rank", min_value=10, max_value=50, value=25, step=5)
    st.markdown("**Required env vars**")
    st.code("OPENAI_API_KEY
ADZUNA_APP_ID
ADZUNA_APP_KEY")
    st.info(
        "LinkedIn scraping is intentionally avoided. For reliable job search results, configure Adzuna keys; DuckDuckGo fallback is best-effort only."
    )

DEFAULT_MODEL = model_name

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_resume = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF, DOCX, TXT",
    )
    expected_roles = st.text_area(
        "Expected roles",
        placeholder="Example: QA Director, Head of Quality Engineering, Testing Transformation Leader",
        height=120,
    )
    expected_designations = st.text_area(
        "Expected designations",
        placeholder="Example: Director, Senior Director, AVP, VP",
        height=100,
    )
    locations = st.text_input(
        "Preferred locations",
        value="United States",
        placeholder="Example: New Jersey, New York, Remote, Dallas",
    )

with col2:
    st.subheader("How it works")
    st.markdown(
        """
1. Parse the uploaded resume
2. Use OpenAI to extract target titles, skills, and search queries
3. Search the web across multiple job sources
4. Rank jobs by fit using the model
5. Show the best matches with reasons
        """
    )

run = st.button("Search Jobs", type="primary", use_container_width=True)

if run:
    try:
        st.write("🚀 Starting job search...")
        if not uploaded_resume:
            st.error("Please upload a resume.")
            st.stop()

        with st.spinner("Reading resume..."):
            resume_text = extract_resume_text(uploaded_resume)
            if not resume_text.strip():
                st.error("Could not extract text from the resume.")
                st.stop()

        with st.spinner("Building job search plan..."):
            st.write("Calling OpenAI to generate search plan...")
            plan = build_search_plan(
                resume_text=resume_text,
                expected_roles=expected_roles,
                expected_designations=expected_designations,
                locations=locations,
            )
            plan = generate_default_queries(plan, locations)

        st.subheader("Candidate Summary")
        st.write(plan.get("candidate_summary", ""))

        a, b = st.columns(2)
        with a:
            st.markdown("**Target Titles**")
            st.write(plan.get("target_titles", []))
        with b:
            st.markdown("**Skill Keywords**")
            st.write(plan.get("skills_keywords", []))

        with st.expander("Generated search queries"):
            st.json(plan.get("search_queries", []))

        with st.spinner("Searching job sources..."):
            st.write("Searching jobs from web...")
            jobs = collect_jobs(plan, max_per_query=6, location=locations)

        if not jobs:
            if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
                st.warning("No job results found. DuckDuckGo fallback is unreliable on Streamlit Cloud. Add ADZUNA_APP_ID and ADZUNA_APP_KEY in Secrets for reliable internet job search.")
            else:
                st.warning("No job results found. Try broader roles or locations.")
            st.stop()

        with st.spinner("Ranking jobs..."):
            st.write("Ranking jobs using AI...")
            ranked_jobs = rank_jobs(
                resume_text=resume_text,
                expected_roles=expected_roles,
                expected_designations=expected_designations,
                jobs=jobs[:max_results],
            )

        st.subheader(f"Top Matches ({len(ranked_jobs)})")
        for job in ranked_jobs:
            with st.container(border=True):
                st.markdown(f"### [{job.get('title', 'Untitled Job')}]({job.get('url', '#')})")
                st.write(f"**Source:** {job.get('source', 'web')}  |  **Fit Score:** {job.get('fit_score', 0)}/100")
                if job.get("display_link"):
                    st.caption(job.get("display_link"))
                if job.get("snippet"):
                    st.write(job.get("snippet"))
                if job.get("reason"):
                    st.write(f"**Why this matches:** {job.get('reason')}")
                highlights = job.get("match_highlights", [])
                if highlights:
                    st.write("**Match highlights:**")
                    for h in highlights:
                        st.write(f"- {h}")

        with st.expander("Raw extracted resume text"):
            st.text_area("Resume text", resume_text[:20000], height=350)

    except Exception as exc:
        st.exception(exc)


# -----------------------------
# Requirements (save separately if needed)
# -----------------------------
# streamlit
# openai
# requests
# PyPDF2
# python-docx

# Run with:
# streamlit run job_agent_streamlit.py
