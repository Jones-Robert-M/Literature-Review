import requests
import time
import os

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

_last_request_time = 0
_request_interval = 1.1 # seconds, slightly more than 1 to be safe

def safe_request(url, params=None, headers=None, backoff=1.0, max_tries=5):
    global _last_request_time

    # Rate limiting: ensure at least _request_interval has passed since last request
    elapsed_time = time.time() - _last_request_time
    if elapsed_time < _request_interval:
        sleep_duration = _request_interval - elapsed_time
        # print(f"Rate limiting: Sleeping for {sleep_duration:.2f} seconds.") # For debugging
        time.sleep(sleep_duration)

    for i in range(max_tries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        _last_request_time = time.time() # Update time after a successful (or attempted) request
        
        if r.status_code == 200:
            return r
        if r.status_code == 400: # Bad Request, likely due to invalid fields
            print(f"Warning: Bad Request for URL: {url} with params {params}. Error: {r.text}")
            r.raise_for_status() # Still raise for 400 as it's a structural issue, but with more info
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
        fields = "title,abstract,authors,year,venue,doi,url,citationCount,references.paperId"
    else:
        # Without an API key, the public API is very restrictive.
        # Requesting minimal fields to increase chances of success.
        # references.paperId is sometimes restricted on public API
        fields = "title,abstract,authors,year,url" 

    params = {"query": query, "limit": limit, "fields": fields}
    try:
        r = safe_request(f"{SEMANTIC_SCHOLAR_API}/paper/search", params=params, headers=headers)
        data = r.json()
        if isinstance(data, dict) and data.get("data"): # Check if data is a dict and "data" key exists and is not None/empty
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
                    # References will be empty if not requested or available without API key
                    "references": [ref['paperId'] for ref in item.get('references', []) if ref.get('paperId')] if api_key else []
                })
        elif isinstance(data, dict) and "error" in data:
            print(f"Semantic Scholar search API returned an error: {data['error']}")
        else:
            print(f"Semantic Scholar search API returned no valid data or malformed response: {data}")
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching from Semantic Scholar search: {e}")
        print("This may be due to a missing or invalid Semantic Scholar API key, or requesting fields not allowed for public access.")
    except Exception as e:
        print(f"An unexpected error occurred during Semantic Scholar search: {e}")
    return papers

def fetch_paper_details(paper_ids):
    papers = []
    headers = {}
    api_key = get_api_key()
    if api_key:
        headers["x-api-key"] = api_key
        fields = "title,abstract,authors,year,venue,doi,url,citationCount"
    else:
        # Without an API key, the public API is very restrictive for batch lookups too.
        fields = "title,abstract,authors,year,url"
    
    chunk_size = 100
    for i in range(0, len(paper_ids), chunk_size):
        chunk = paper_ids[i:i + chunk_size]
        # Skip if chunk is empty (e.g., all paper_ids were invalid)
        if not chunk:
            continue
        params = {"fields": fields}
        try:
            # Use POST for batch API
            r = requests.post(f"{SEMANTIC_SCHOLAR_API}/paper/batch", headers=headers, json={"ids": chunk})
            _last_request_time = time.time() # Update time after a successful (or attempted) request
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                for item in data:
                    if item and item.get("paperId"): # Check if item is not None and has a paperId
                        papers.append({
                            "id": item.get("paperId"),
                            "title": item.get("title", ""),
                            "abstract": item.get("abstract", "") or "",
                            "authors": ";".join([a.get("name","") for a in item.get("authors", [])]),
                            "published": str(item.get("year","")),
                            "source": "SemanticScholar",
                            "doi": item.get("doi","") or "", # Will be empty if not requested or available
                            "url": item.get("url",""),
                            "citationCount": item.get("citationCount", 0), # Will be empty if not requested or available
                        })
                    else:
                        print(f"Warning: Skipping paper in batch due to missing paperId or null item: {item}")
            else:
                print(f"Semantic Scholar batch API returned no valid list data or malformed response: {data}")

        except requests.exceptions.HTTPError as e:
            print(f"Error fetching paper details for a batch (first ID: {chunk[0] if chunk else 'N/A'}): {e}")
            print("This may be due to a missing or invalid Semantic Scholar API key, or requesting fields not allowed for public access.")
        except Exception as e:
            print(f"An unexpected error occurred while fetching paper details for a batch (first ID: {chunk[0] if chunk else 'N/A'}): {e}")
    return papers
