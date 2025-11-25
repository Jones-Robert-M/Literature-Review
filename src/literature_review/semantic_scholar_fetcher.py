import requests
import time
import os

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

def safe_request(url, params=None, headers=None, backoff=1.0, max_tries=5):
    for i in range(max_tries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        time.sleep(backoff * (2 ** i))
    r.raise_for_status()

def get_api_key():
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def fetch_semanticscholar(query, limit=100):
    papers = []
    headers = {}
    api_key = get_api_key()
    if api_key:
        headers["x-api-key"] = api_key

    params = {"query": query, "limit": limit, "fields": "title,abstract,authors,year,venue,doi,url,citationCount,references.paperId"}
    r = safe_request(f"{SEMANTIC_SCHOLAR_API}/paper/search", params=params, headers=headers)
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
            "citationCount": item.get("citationCount", 0),
            "references": [ref['paperId'] for ref in item.get('references', []) if ref.get('paperId')]
        })
    return papers

def fetch_paper_details(paper_ids):
    papers = []
    headers = {}
    api_key = get_api_key()
    if api_key:
        headers["x-api-key"] = api_key
    
    # The API has a limit on the number of paper IDs that can be requested at once.
    # I'll split the list of paper IDs into chunks of 100.
    chunk_size = 100
    for i in range(0, len(paper_ids), chunk_size):
        chunk = paper_ids[i:i + chunk_size]
        params = {"fields": "title,abstract,authors,year,venue,doi,url,citationCount"}
        r = requests.post(f"{SEMANTIC_SCHOLAR_API}/paper/batch", params=params, headers=headers, json={"ids": chunk})
        r.raise_for_status()
        data = r.json()
        for item in data:
            papers.append({
                "id": item.get("paperId"),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "") or "",
                "authors": ";".join([a.get("name","") for a in item.get("authors", [])]),
                "published": str(item.get("year","")),
                "source": "SemanticScholar",
                "doi": item.get("doi","") or "",
                "url": item.get("url",""),
                "citationCount": item.get("citationCount", 0),
            })
    return papers