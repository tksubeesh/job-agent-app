import html
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Job Search Agent",
    page_icon="🔎",
    layout="wide",
)

REQUEST_TIMEOUT = 20
DEFAULT_MAX_RESULTS = 25
MAX_QUERIES = 18
USER_AGENT = "job-search-agent/2.0"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}


# =========================
# Data Model
# =========================
@dataclass
class JobRecord:
    source: str
    source_id: str
    title: str
    company: str
    location: str
    remote_type: str
    description: str
    url: str
    posted_at: str
    salary: str
    employment_type: str
    tags: List[str]
    raw: Dict[str, Any]
    match_score: float = 0.0
    match_explanation: str = ""


# =========================
# Helpers
# =========================
def clean_html(text: Any) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def contains_any(text: str, phrases: List[str]) -> bool:
    text_n = normalize_text(text)
    return any(p in text_n for p in phrases)


def safe_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def parse_datetime_like(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        text = text.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        pass

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def days_ago_text(value: Any) -> str:
    dt = parse_datetime_like(value)
    if not dt:
        return "Unknown"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    days = delta.days
    if days <= 0:
        hours = max(int(delta.total_seconds() // 3600), 0)
        return f"{hours}h ago" if hours > 0 else "Today"
    if days == 1:
        return "1 day ago"
    if days < 30:
        return f"{days} days ago"
    months = max(days // 30, 1)
    return f"{months} mo ago"


def compact_list(items: List[str], max_items: int = 6) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = re.sub(r"\s+", " ", str(item or "").strip())
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:max_items]


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Query Generation
# =========================
def build_title_pool(user_titles: List[str]) -> List[str]:
    defaults = [
        "Director Quality Engineering",
        "Director QA",
        "Director Testing",
        "Head of QA",
        "Director Test Automation",
        "Director Quality Transformation",
        "Delivery Director",
        "Program Delivery Director",
        "Director Digital Delivery",
        "Quality Engineering Leader",
        "Testing Transformation Leader",
        "Client Partner",
        "Engagement Director",
        "Business Relationship Manager",
    ]
    merged = compact_list(user_titles + defaults, max_items=30)
    return merged


def generate_search_queries(profile: Dict[str, Any]) -> List[str]:
    titles = build_title_pool(profile["preferred_titles"])
    location = profile["location"]
    include_remote = profile["include_remote"]
    skills = compact_list(profile["skills"] + [
        "quality engineering",
        "test automation",
        "QA transformation",
        "digital transformation",
        "program delivery",
        "enterprise delivery",
        "stakeholder management",
        "portfolio delivery",
        "AI testing",
        "GenAI",
    ], max_items=20)

    industries = compact_list(profile["industries"] + [
        "telecom",
        "technology",
        "enterprise",
        "digital",
    ], max_items=12)

    queries: List[str] = []

    for title in titles[:12]:
        queries.append(title)

        if location:
            queries.append(f"{title} {location}")

        if include_remote:
            queries.append(f"{title} remote")
            queries.append(f"{title} hybrid")

        for skill in skills[:3]:
            queries.append(f"{title} {skill}")

        for industry in industries[:2]:
            queries.append(f"{title} {industry}")

    # Add broader adjacency
    adjacency = [
        "Director Delivery",
        "Transformation Director",
        "Quality Leader",
        "Engineering Program Director",
        "QA Leader",
        "Enterprise Delivery Leader",
    ]
    queries.extend(adjacency)

    # Clean + dedupe + cap
    final_queries: List[str] = []
    seen = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            continue
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        final_queries.append(q)

    return final_queries[:MAX_QUERIES]


def broaden_query(query: str) -> List[str]:
    variants = [query]

    replacements = [
        ("Director Quality Engineering", "Director QA"),
        ("Director QA", "Head of QA"),
        ("Director Testing", "QA Leader"),
        ("Program Delivery Director", "Delivery Director"),
        ("Director Quality Transformation", "Transformation Director"),
        ("Client Partner", "Engagement Director"),
        ("Business Relationship Manager", "Client Partner"),
        ("Quality Engineering Leader", "Quality Leader"),
    ]

    for old, new in replacements:
        if old.lower() in query.lower():
            variants.append(re.sub(re.escape(old), new, query, flags=re.I))

    stripped = re.sub(r"\b(remote|hybrid)\b", " ", query, flags=re.I)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped and stripped.lower() != query.lower():
        variants.append(stripped)

    generic_titles = [
        "Director QA",
        "Head of QA",
        "Delivery Director",
        "Program Director",
        "Client Partner",
        "Transformation Director",
    ]
    for title in generic_titles:
        if title.lower() in query.lower():
            variants.append(title)

    out = []
    seen = set()
    for item in variants:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# =========================
# Source Adapters
# =========================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_remotive(query: str) -> List[JobRecord]:
    """
    Official public endpoint supports:
    GET https://remotive.com/api/remote-jobs?search=...
    """
    payload = safe_get(
        "https://remotive.com/api/remote-jobs",
        params={"search": query},
    )
    if not payload:
        return []

    jobs = payload.get("jobs", [])
    results: List[JobRecord] = []

    for item in jobs:
        title = item.get("title", "")
        company = item.get("company_name", "")
        location = item.get("candidate_required_location", "") or "Remote"
        description = clean_html(item.get("description", ""))
        url = item.get("url", "")
        job_id = str(item.get("id", ""))
        posted_at = item.get("publication_date", "")
        salary = item.get("salary", "")
        category = item.get("category", "")
        employment_type = str(item.get("job_type", "")).replace("_", " ").title()

        tags = compact_list([category, employment_type, "Remote"])

        results.append(
            JobRecord(
                source="Remotive",
                source_id=job_id or f"remotive-{hash(url)}",
                title=title,
                company=company,
                location=location,
                remote_type="Remote",
                description=description,
                url=url,
                posted_at=posted_at,
                salary=salary,
                employment_type=employment_type,
                tags=tags,
                raw=item,
            )
        )

    return results


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_arbeitnow_pages(page_count: int = 6) -> List[JobRecord]:
    """
    Arbeitnow free API is no-key and page based.
    We fetch pages broadly and do our own ranking locally.
    """
    results: List[JobRecord] = []

    for page in range(1, page_count + 1):
        payload = safe_get(
            "https://www.arbeitnow.com/api/job-board-api",
            params={"page": page},
        )
        if not payload:
            continue

        data = payload.get("data", [])
        if not data:
            continue

        for item in data:
            title = item.get("title", "")
            company = item.get("company_name", "") or item.get("company", "")
            location_parts = item.get("location", []) or []
            if isinstance(location_parts, list):
                location = ", ".join([str(x) for x in location_parts if x]) or "Germany"
            else:
                location = str(location_parts or "Germany")

            remote_flag = item.get("remote", False)
            remote_type = "Remote/Hybrid" if remote_flag else "Onsite/Hybrid"

            description = clean_html(item.get("description", ""))
            url = item.get("url", "")
            job_id = str(item.get("slug", "")) or str(item.get("id", "")) or f"arbeitnow-{hash(url)}"
            posted_at = item.get("created_at", "") or item.get("updated_at", "")
            salary = item.get("salary", "") or ""
            employment_type = ", ".join(item.get("job_types", []) or [])
            tags = compact_list(
                (item.get("tags", []) or [])
                + (item.get("job_types", []) or [])
                + (["Visa Sponsorship"] if item.get("visa_sponsorship") else [])
                + (["Remote"] if remote_flag else [])
            )

            results.append(
                JobRecord(
                    source="Arbeitnow",
                    source_id=job_id,
                    title=title,
                    company=company,
                    location=location,
                    remote_type=remote_type,
                    description=description,
                    url=url,
                    posted_at=posted_at,
                    salary=salary,
                    employment_type=employment_type,
                    tags=tags,
                    raw=item,
                )
            )

    return results


# =========================
# Ranking / Filtering
# =========================
def normalize_for_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def make_job_key(job: JobRecord) -> str:
    title = normalize_for_key(job.title)
    company = normalize_for_key(job.company)
    location = normalize_for_key(job.location)
    return f"{title}|{company}|{location}"


def dedupe_jobs(jobs: List[JobRecord]) -> List[JobRecord]:
    best_by_key: Dict[str, JobRecord] = {}

    for job in jobs:
        key = make_job_key(job)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = job
        else:
            # Keep the one with richer description / url
            existing_len = len(existing.description or "")
            current_len = len(job.description or "")
            if current_len > existing_len:
                best_by_key[key] = job

    return list(best_by_key.values())


def score_job(job: JobRecord, profile: Dict[str, Any], strictness: str) -> Tuple[float, List[str]]:
    title = normalize_text(job.title)
    description = normalize_text(job.description)
    company = normalize_text(job.company)
    location = normalize_text(job.location)
    all_text = " ".join([title, description, company, location, " ".join([normalize_text(t) for t in job.tags])])

    score = 0.0
    reasons: List[str] = []

    # 1) Strong titles
    strong_titles = [
        "director quality engineering",
        "director qa",
        "director testing",
        "head of qa",
        "director test automation",
        "director quality transformation",
        "delivery director",
        "program delivery director",
        "director digital delivery",
        "quality engineering leader",
        "testing transformation leader",
        "client partner",
        "engagement director",
        "business relationship manager",
    ]
    for phrase in strong_titles:
        if phrase in title:
            score += 36
            reasons.append("strong title fit")

    # 2) Leadership signals
    leadership_terms = [
        "director", "head", "leader", "lead", "portfolio", "stakeholder",
        "governance", "delivery", "transformation", "executive", "program",
    ]
    leadership_hits = 0
    for term in leadership_terms:
        if term in title:
            score += 7
            leadership_hits += 1
        if term in description:
            score += 2.2
    if leadership_hits >= 2:
        reasons.append("leadership alignment")

    # 3) QE / Testing / Delivery fit
    fit_terms = [
        "quality engineering",
        "quality assurance",
        "qa",
        "testing",
        "test automation",
        "automation",
        "sdet",
        "enterprise delivery",
        "program delivery",
        "digital transformation",
        "release management",
        "agile",
        "safe",
        "telecom",
        "genai",
        "gen ai",
        "ai testing",
    ]
    fit_hits = 0
    for term in fit_terms:
        if term in title:
            score += 11
            fit_hits += 1
        if term in description:
            score += 3.5
            fit_hits += 1
    if fit_hits >= 2:
        reasons.append("functional fit")

    # 4) Preferred titles
    user_titles = [normalize_text(x) for x in profile["preferred_titles"]]
    for ut in user_titles:
        if ut and ut in title:
            score += 18
            reasons.append("matches preferred title")
            break

    # 5) Skills / industries
    for skill in [normalize_text(x) for x in profile["skills"]]:
        if skill and skill in all_text:
            score += 5
    for industry in [normalize_text(x) for x in profile["industries"]]:
        if industry and industry in all_text:
            score += 4

    # 6) Location / remote fit
    preferred_location = normalize_text(profile["location"])
    if preferred_location and preferred_location in location:
        score += 12
        reasons.append("location fit")

    if profile["include_remote"]:
        if contains_any(job.remote_type, ["remote"]) or contains_any(location, ["remote", "worldwide", "usa only"]):
            score += 10
            reasons.append("remote-friendly")

    # 7) Seniority penalty
    penalty_terms = [
        "intern", "junior", "entry level", "coordinator", "analyst",
        "technician", "manual tester", "qa tester", "sdet i", "sdet ii",
        "associate", "specialist",
    ]
    for term in penalty_terms:
        if term in title:
            score -= 30

    # 8) Exclude keywords
    for term in [normalize_text(x) for x in profile["exclude_keywords"]]:
        if term and term in all_text:
            score -= 18

    # 9) Required keywords
    required_terms = [normalize_text(x) for x in profile["must_have_keywords"]]
    if required_terms:
        matched_required = sum(1 for term in required_terms if term and term in all_text)
        score += matched_required * 6
        if strictness == "Strict" and matched_required == 0:
            score -= 25

    # 10) Recency boost
    dt = parse_datetime_like(job.posted_at)
    if dt:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_old = max((now - dt).days, 0)
        if days_old <= 3:
            score += 8
        elif days_old <= 7:
            score += 5
        elif days_old <= 30:
            score += 2

    # Strictness tuning
    if strictness == "Strict":
        if "director" in title or "head" in title:
            score += 8
        if "manager" in title:
            score -= 6
    elif strictness == "Broad":
        if "manager" in title:
            score += 4
        if "lead" in title:
            score += 3

    # Explanation cleanup
    reasons = compact_list(reasons, max_items=4)
    if not reasons:
        reasons = ["adjacent leadership relevance"]

    return round(score, 1), reasons


def finalize_results(raw_jobs: List[JobRecord], profile: Dict[str, Any], strictness: str, top_n: int) -> Dict[str, Any]:
    deduped = dedupe_jobs(raw_jobs)

    for job in deduped:
        score, reasons = score_job(job, profile, strictness)
        job.match_score = score
        job.match_explanation = ", ".join(reasons)

    ranked = sorted(deduped, key=lambda x: x.match_score, reverse=True)

    threshold_map = {
        "Strict": 32,
        "Balanced": 24,
        "Broad": 16,
    }
    threshold = threshold_map.get(strictness, 24)

    filtered = [job for job in ranked if job.match_score >= threshold]

    if len(filtered) < top_n:
        filtered = ranked[:top_n]

    return {
        "raw_count": len(raw_jobs),
        "deduped_count": len(deduped),
        "final_count": len(filtered[:top_n]),
        "jobs": filtered[:top_n],
        "threshold": threshold,
    }


# =========================
# Search Orchestration
# =========================
def run_search(profile: Dict[str, Any], strictness: str, top_n: int) -> Dict[str, Any]:
    queries = generate_search_queries(profile)

    all_jobs: List[JobRecord] = []
    query_log: List[Tuple[str, int]] = []

    # Broad source: Arbeitnow pages once
    arbeitnow_jobs = fetch_arbeitnow_pages(page_count=6)
    all_jobs.extend(arbeitnow_jobs)
    query_log.append(("Arbeitnow broad pages", len(arbeitnow_jobs)))

    # Query-based source: Remotive
    for query in queries:
        jobs = fetch_remotive(query)
        all_jobs.extend(jobs)
        query_log.append((f"Remotive: {query}", len(jobs)))

    # Fallback broadening if too thin
    if len(all_jobs) < 80:
        expanded: List[str] = []
        for q in queries[:8]:
            expanded.extend(broaden_query(q))

        expanded_clean: List[str] = []
        seen = set()
        for q in expanded:
            k = q.lower()
            if k not in seen:
                seen.add(k)
                expanded_clean.append(q)

        for query in expanded_clean[:10]:
            jobs = fetch_remotive(query)
            all_jobs.extend(jobs)
            query_log.append((f"Fallback Remotive: {query}", len(jobs)))

    final = finalize_results(all_jobs, profile, strictness, top_n)
    final["queries"] = queries
    final["query_log"] = query_log
    return final


# =========================
# UI
# =========================
def default_profile() -> Dict[str, Any]:
    return {
        "preferred_titles": [
            "Director Quality Engineering",
            "Director QA",
            "Director Testing",
            "Delivery Director",
            "Client Partner",
        ],
        "skills": [
            "quality engineering",
            "test automation",
            "program delivery",
            "digital transformation",
            "GenAI",
        ],
        "industries": ["telecom", "technology"],
        "location": "New Jersey",
        "include_remote": True,
        "must_have_keywords": [],
        "exclude_keywords": ["intern", "junior", "entry level"],
    }


def parse_multiline_or_csv(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\n,;]+", text)
    return compact_list([p.strip() for p in parts if p.strip()], max_items=50)


def render_job_card(job: JobRecord, rank: int) -> None:
    posted_label = days_ago_text(job.posted_at)
    tags = compact_list(job.tags, max_items=8)

    with st.container(border=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(f"### {rank}. [{job.title}]({job.url})")
            st.write(f"**{job.company}**")
            st.write(f"{job.location} • {job.remote_type}")
            st.write(f"**Why it matches:** {job.match_explanation}")

        with col2:
            st.metric("Score", f"{job.match_score:.1f}")
            st.caption(posted_label)

        meta = []
        if job.salary:
            meta.append(f"Salary: {job.salary}")
        if job.employment_type:
            meta.append(f"Type: {job.employment_type}")
        if job.source:
            meta.append(f"Source: {job.source}")
        if meta:
            st.caption(" | ".join(meta))

        if tags:
            st.caption("Tags: " + " • ".join(tags))

        snippet = job.description[:650].strip()
        if snippet:
            st.write(snippet + ("..." if len(job.description) > 650 else ""))


def jobs_to_dataframe(jobs: List[JobRecord]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        rows.append({
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "remote_type": job.remote_type,
            "score": job.match_score,
            "why_match": job.match_explanation,
            "posted": days_ago_text(job.posted_at),
            "salary": job.salary,
            "employment_type": job.employment_type,
            "source": job.source,
            "url": job.url,
        })
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
st.title("🔎 Job Search Agent")
st.caption("Broader retrieval + better ranking for senior QA / delivery / transformation roles")

with st.sidebar:
    st.header("Search Profile")

    defaults = default_profile()

    preferred_titles_text = st.text_area(
        "Preferred titles",
        value="\n".join(defaults["preferred_titles"]),
        height=140,
        help="One per line or comma-separated.",
    )

    skills_text = st.text_area(
        "Core skills / themes",
        value=", ".join(defaults["skills"]),
        height=100,
    )

    industries_text = st.text_input(
        "Industries",
        value=", ".join(defaults["industries"]),
    )

    location = st.text_input("Preferred location", value=defaults["location"])
    include_remote = st.checkbox("Include remote-friendly jobs", value=True)

    must_have_keywords_text = st.text_input(
        "Boost if contains these keywords",
        value="",
        help="Comma-separated. Example: telecom, stakeholder, governance",
    )

    exclude_keywords_text = st.text_input(
        "Penalize if contains these keywords",
        value=", ".join(defaults["exclude_keywords"]),
    )

    strictness = st.selectbox("Search strictness", ["Balanced", "Strict", "Broad"], index=0)
    top_n = st.slider("Results to show", min_value=10, max_value=50, value=25, step=5)

    run_btn = st.button("Run search", type="primary", use_container_width=True)

profile = {
    "preferred_titles": parse_multiline_or_csv(preferred_titles_text),
    "skills": parse_multiline_or_csv(skills_text),
    "industries": parse_multiline_or_csv(industries_text),
    "location": location.strip(),
    "include_remote": include_remote,
    "must_have_keywords": parse_multiline_or_csv(must_have_keywords_text),
    "exclude_keywords": parse_multiline_or_csv(exclude_keywords_text),
}

if run_btn:
    with st.spinner("Searching across sources, widening queries, ranking results..."):
        final = run_search(profile, strictness, top_n)

    jobs = final["jobs"]
    df = jobs_to_dataframe(jobs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw jobs", final["raw_count"])
    c2.metric("Unique jobs", final["deduped_count"])
    c3.metric("Returned", final["final_count"])
    c4.metric("Threshold", final["threshold"])

    with st.expander("Queries tried", expanded=False):
        st.write(final["queries"])
        log_df = pd.DataFrame(final["query_log"], columns=["query", "jobs_found"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    if df.empty:
        st.warning("No jobs came back after ranking. Try Broad mode and remove narrow keywords.")
    else:
        st.subheader("Top matches")

        for idx, job in enumerate(jobs, start=1):
            render_job_card(job, idx)

        st.subheader("Export")
        st.download_button(
            "Download results as CSV",
            data=to_csv_download(df),
            file_name="job_search_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Tabular view", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info(
        "Set your profile in the sidebar and click **Run search**. "
        "Balanced mode is the safest default; Broad mode is better when the market is thin."
    )