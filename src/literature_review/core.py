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
import Levenshtein
from .plotting_utils import set_favourite_plot_params, apply_favourite_figure_params
from itertools import combinations

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

def dedupe_papers(papers, title_thresh=5):
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
    comp_gen = girvan_newman(G)
    communities = None
    try:
        for com in comp_gen:
            communities = [list(c) for c in com]
            if len(communities) >= max_communities:
                break
    except Exception:
        communities = [list(G.nodes())]
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
        summaries[cid] = {"n_papers": len(idx), "top_terms": top, "paper_indices": [int(i) for i in idx]}
    return summaries

def sentiment_analysis(texts):
    nlp = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    res = nlp(texts, truncation=True)
    scores = []
    for r in res:
        s = r['score'] if r['label']=='POSITIVE' else -r['score']
        scores.append(s)
    return scores

def summarize_sentiment(scores):
    avg_score = np.mean(scores)
    positive_count = np.sum(np.array(scores) > 0.5)
    negative_count = np.sum(np.array(scores) < -0.5)
    neutral_count = len(scores) - positive_count - negative_count
    summary = f"The average sentiment score across the corpus is {avg_score:.2f}.\n"
    summary += f"There are {positive_count} positive, {negative_count} negative, and {neutral_count} neutral papers.\n"
    if avg_score > 0.2:
        summary += "Overall, the sentiment of the corpus is positive."
    elif avg_score < -0.2:
        summary += "Overall, the sentiment of the corpus is negative."
    else:
        summary += "Overall, the sentiment of the corpus is neutral."
    return summary

def summarize_text_local(text):
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def rank_papers(df, embeddings, query_embedding=None, sim=None):
    n = len(df)
    G = nx.from_numpy_array(sim) if sim is not None else nx.Graph()
    scores = np.zeros(n)
    if len(G)>0:
        pr = nx.pagerank(G, weight='weight')
        for i in range(n):
            scores[i] += pr.get(i, 0.0)
    if 'citationCount' in df.columns:
        c = df['citationCount'].fillna(0).to_numpy().astype(float)
        if c.max()>0:
            scores += (c / c.max())
    if query_embedding is not None:
        qsim = cosine_similarity([query_embedding], embeddings).ravel()
        scores += qsim
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

def calculate_reference_overlap(papers):
    """Calculates the reference overlap between papers."""
    paper_references = {p['id']: set(p.get('references', [])) for p in papers if 'id' in p}
    overlaps = []
    for p1, p2 in combinations(paper_references.keys(), 2):
        if p1 in paper_references and p2 in paper_references:
            overlap = len(paper_references[p1].intersection(paper_references[p2]))
            if overlap > 0:
                overlaps.append(overlap)
    return overlaps

def plot_reference_overlap(overlaps, output_path):
    """Plots the distribution of reference overlaps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(overlaps, bins=20)
    ax.set_title('Reference Overlap Distribution', fontsize=16, fontweight='bold')
    ax = set_favourite_plot_params(ax, x_title='Number of Shared References', y_title='Number of Paper Pairs')
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path)
    plt.close()

def visualize_graph(G, labels, cluster_summaries, output_path):
    """Visualizes a graph with nodes colored by cluster and a legend with cluster terms."""
    fig, ax = plt.subplots(figsize=(16, 16))
    pos = nx.kamada_kawai_layout(G)
    colors = [labels.get(node, 0) for node in G.nodes()]
    node_sizes = [cluster_summaries.get(labels.get(node, 0), {}).get('n_papers', 1) * 20 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.plasma, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    legend_labels = {}
    for cluster_id, summary in cluster_summaries.items():
        top_terms = ", ".join(summary['top_terms'][:3])
        legend_labels[cluster_id] = f"Cluster {cluster_id}: {top_terms}"
    from matplotlib.lines import Line2D
    cmap = plt.cm.plasma
    custom_lines = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(i/len(legend_labels)), markersize=15) for i in range(len(legend_labels))]
    ax.legend(custom_lines, [legend_labels.get(i, '') for i in range(len(legend_labels))], title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.set_title("Paper Similarity Network", fontsize=24, fontweight='bold')
    ax = set_favourite_plot_params(ax, x_title='', y_title='', spine_width=0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path, dpi=300)
    plt.close()