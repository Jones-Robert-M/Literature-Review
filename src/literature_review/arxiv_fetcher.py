import arxiv

def fetch_arxiv(query, max_results=200):
    """
    Fetches papers from arXiv.
    """
    results = []
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    for r in search.results():
        results.append({
            "id": r.entry_id,
            "title": r.title,
            "abstract": r.summary,
            "authors": ", ".join([a.name for a in r.authors]),
            "published": r.published.isoformat(),
            "source": "arXiv",
            "doi": r.doi or "",
            "url": r.pdf_url or r.entry_id
        })
    return results
