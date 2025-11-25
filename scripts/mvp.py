"""
lit_review.py — MVP autonomous literature review (arXiv + Semantic Scholar)
Produces: papers.csv, clusters.csv, wordcloud.png, cluster_summaries.json
Dependencies (pip): arxiv, sentence-transformers, scikit-learn, networkx, pandas,
                   numpy, matplotlib, wordcloud, transformers, requests, python-Levenshtein
Run: python lit_review.py --query "information architecture OR \"sensor fusion\"" --max 250
"""

import argparse
import os
import time
import json
from collections import defaultdict
from pprint import pprint

import arxiv
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from networkx.algorithms.community import girvan_newman
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
import Levenshtein  # for fuzzy dedupe

import os

# src/literature_pipeline/scopus_client.py
import requests
from ..configs import SCOPUS_API_KEY

BASE_URL = "https://api.elsevier.com/content/search/scopus"

def scopus_search(query: str, count: int = 25):
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    params  = {"query": query, "count": count}

    r = requests.get(BASE_URL, headers=headers, params=params)
    r.raise_for_status()
    return r.json()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

SCOPUS_API_KEY = get_env("SCOPUS_API_KEY")
SCOPUS_INST_TOKEN = get_env("SCOPUS_INST_TOKEN")
ELSEVIER_API_KEY = get_env("ELSEVIER_API_KEY")

# -----------------------------
# Config
# -----------------------------
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_DIR = "lr_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def safe_request(url, params=None, headers=None, backoff=1.0, max_tries=5):
    for i in range(max_tries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        time.sleep(backoff * (2 ** i))
    r.raise_for_status()

def dedupe_papers(papers, title_thresh=5):
    # simple dedupe by Levenshtein distance on titles
    out = []
    titles = []
    for p in papers:
        t = p.get("title","").lower().strip()
        keep = True
        for t2 in titles:
            if Levenshtein.distance(t, t2) <= title_thresh:
                keep = False
                break
        if keep:
            out.append(p)
            titles.append(t)
    return out

# -----------------------------
# Harvesters
# -----------------------------
def fetch_arxiv(query, max_results=200):
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

def fetch_semanticscholar(query, limit=100):
    # fallback search to enrich metadata (citations) — limited fields for public API
    papers = []
    params = {"query": query, "limit": limit, "fields": "title,abstract,authors,year,venue,doi,url,citationCount"}
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

# -----------------------------
# Core pipeline
# -----------------------------
def build_corpus(papers):
    df = pd.DataFrame(papers)
    df['text'] = (df['title'].fillna('') + ". " + df['abstract'].fillna('')).str.replace("\n"," ", regex=False)
    return df

def embed_texts(texts, model_name=EMBED_MODEL, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    return np.array(embeddings)

def build_similarity_graph(embeddings, top_k=8, thresh=0.55):
    n = embeddings.shape[0]
    sim = cosine_similarity(embeddings)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        # connect to top_k highest similarities above thresh (exclude self)
        idx = np.argsort(-sim[i])
        added = 0
        for j in idx:
            if j == i: continue
            if sim[i,j] < thresh and added>=top_k: break
            if sim[i,j] >= thresh or added < top_k:
                G.add_edge(i,j,weight=float(sim[i,j]))
                added += 1
    return G, sim

def run_girvan_newman_graph(G, max_communities=8):
    # Girvan-Newman yields hierarchical splits; take first split that gives at most max_communities
    comp_gen = girvan_newman(G)
    communities = None
    try:
        for com in comp_gen:
            communities = [list(c) for c in com]
            if len(communities) >= max_communities:
                break
    except Exception:
        communities = [list(G.nodes())]
    # label nodes
    labels = {}
    for i, c in enumerate(communities):
        for node in c:
            labels[node] = i
    return labels, communities

def cluster_summaries(df, labels, top_k_terms=15):
    summaries = {}
    for cid in sorted(set(labels.values())):
        idx = [i for i,v in labels.items() if v==cid]
        text = " ".join(df.iloc[idx]['text'].tolist())
        tf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
        X = tf.fit_transform([text])
        terms = np.array(tf.get_feature_names_out())
        scores = X.toarray().ravel()
        top = terms[np.argsort(-scores)][:top_k_terms].tolist()
        summaries[cid] = {"n_papers": len(idx), "top_terms": top, "paper_indices": idx}
    return summaries

def sentiment_analysis(texts):
    nlp = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    res = nlp(texts, truncation=True)
    # convert to simple score: POSITIVE->+1 * score, NEGATIVE->-1 * score
    scores = []
    for r in res:
        s = r['score'] if r['label']=="POSITIVE" else -r['score']
        scores.append(s)
    return scores

def rank_papers(df, embeddings, query_embedding=None, sim=None):
    n = len(df)
    G = nx.from_numpy_array(sim) if sim is not None else nx.Graph()
    scores = np.zeros(n)
    # centrality measures
    if len(G)>0:
        pr = nx.pagerank(G, weight='weight')
        for i in range(n):
            scores[i] += pr.get(i, 0.0)
    # citations if available (normalised)
    if 'citationCount' in df.columns:
        c = df['citationCount'].fillna(0).to_numpy().astype(float)
        if c.max()>0:
            scores += (c / (c.max()))
    # relevance to query
    if query_embedding is not None:
        qsim = cosine_similarity([query_embedding], embeddings).ravel()
        scores += qsim
    # recency boost (more recent papers get slight boost)
    # published may be year or iso date — try year parse
    years = df['published'].fillna("")
    year_arr = []
    for y in years:
        try:
            yv = int(str(y)[:4])
        except Exception:
            yv = 1970
        year_arr.append(yv)
    year_arr = np.array(year_arr)
    if year_arr.max()>0:
        scores += (year_arr - year_arr.min()) / (year_arr.max() - year_arr.min() + 1e-9) * 0.5
    df['score'] = scores
    df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df_sorted

# -----------------------------
# Main CLI
# -----------------------------
def main(args):
    # 1. Harvest from arXiv
    print("Fetching arXiv...")
    arx = fetch_arxiv(args.query, max_results=args.max)
    print(f"Got {len(arx)} arXiv records")

    # 2. Harvest from Semantic Scholar (search)
    print("Fetching Semantic Scholar search results...")
    ss = fetch_semanticscholar(args.query, limit=min(200, args.max))
    print(f"Got {len(ss)} Semantic Scholar records")

    # merge and dedupe
    raw = arx + ss
    raw = dedupe_papers(raw)
    print(f"After dedupe: {len(raw)} papers")

    # build dataframe
    df = build_corpus(raw)
    if df.empty:
        print("No papers found. Exiting.")
        return

    # embeddings
    print("Embedding texts (may take a minute)...")
    texts = df['text'].tolist()
    embeddings = embed_texts(texts)

    # build graph
    print("Building similarity graph...")
    G, sim = build_similarity_graph(embeddings, top_k=6, thresh=0.55)

    # community detection
    print("Running Girvan-Newman...")
    labels, communities = run_girvan_newman_graph(G, max_communities=min(8, len(df)))
    df['cluster'] = df.index.map(lambda i: labels.get(i, -1))

    # cluster summaries
    print("Producing cluster summaries...")
    summaries = cluster_summaries(df, labels)

    # sentiment (on abstracts)
    print("Running sentiment analysis on abstracts...")
    s_texts = df['abstract'].fillna("").replace("", "neutral")
    sent_scores = sentiment_analysis(s_texts.tolist())
    df['sentiment'] = sent_scores

    # rank papers
    print("Ranking papers...")
    # query embedding (use same model)
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([args.query], normalize_embeddings=True)[0]
    ranked = rank_papers(df.copy(), embeddings, query_embedding=q_emb, sim=sim)

    # outputs
    print("Saving outputs...")
    df.to_csv("papers.csv", index=False)
    ranked.to_csv("ranked_papers.csv", index=False)
    with open("cluster_summaries.json", "w") as fh:
        json.dump(summaries, fh, indent=2)
    # simple wordcloud for corpus
    all_text = " ".join(df['text'].tolist())
    wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
    plt.figure(figsize=(16,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("wordcloud.png", dpi=150)
    print("Done. Outputs: papers.csv, ranked_papers.csv, cluster_summaries.json, wordcloud.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--max", type=int, default=200)
    args = p.parse_args()
    main(args)
