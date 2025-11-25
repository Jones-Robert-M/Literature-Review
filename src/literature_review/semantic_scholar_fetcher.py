import requests
import time

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

def safe_request(url, params=None, headers=None, backoff=1.0, max_tries=5):
    for i in range(max_tries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        time.sleep(backoff * (2 ** i))
    r.raise_for_status()

def fetch_semanticscholar(query, limit=100):
    # fallback search to enrich metadata (citations) — limited fields for public API
    papers = []
    params = {"query": query, "limit": limit, "fields": "title,abstract"}
    r = safe_request(SEMANTIC_SCHOLAR_API, params=params)
    data = r.json()
    for item in data.get("data", []):
        papers.append({
            "id": item.get("paperId"),
            "title": item.get("title", ""),
            "abstract": item.get("abstract", "") or "",
            "authors": ";".join([a.get("name","") for a in item.get("authors", [])]),
            "published": str(item.get("year","")),
            "source": "SemanticScholar",
            "doi": item.get("doi","") or "",
            "url": item.get("url",""),
            "citationCount": item.get("citationCount", 0)
        })
    return papers
