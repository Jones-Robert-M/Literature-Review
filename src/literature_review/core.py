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

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

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
        summaries[cid] = {"n_papers": len(idx), "top_terms": top, "paper_indices": [int(i) for i in idx]}
    return summaries

def sentiment_analysis(texts):
    nlp = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    res = nlp(texts, truncation=True)
    # convert to simple score: POSITIVE->+1 * score, NEGATIVE->-1 * score
    scores = []
    for r in res:
        s = r['score'] if r['label']=='POSITIVE' else -r['score']
        scores.append(s)
    return scores

def summarize_sentiment(scores):
    """Generates a human-readable summary of sentiment scores."""
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
    """Summarizes a text using a local model."""
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

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

def visualize_graph(G, labels, cluster_summaries, output_path):
    """Visualizes a graph with nodes colored by cluster and a legend with cluster terms."""
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.1, iterations=20)
    
    # Create a color map from cluster labels
    colors = [labels.get(node, 0) for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Create a legend for the clusters
    legend_labels = {}
    for cluster_id, summary in cluster_summaries.items():
        top_terms = ", ".join(summary['top_terms'][:3])
        legend_labels[cluster_id] = f"Cluster {cluster_id}: {top_terms}"

    # Add the legend to the plot
    # We need to create proxy artists for the legend
    from matplotlib.lines import Line2D
    cmap = plt.cm.viridis
    custom_lines = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(i/len(legend_labels)), markersize=10) for i in range(len(legend_labels))]

    plt.legend(custom_lines, [legend_labels[i] for i in range(len(legend_labels))], title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Paper Similarity Network")
    plt.tight_layout()
    plt.savefig(output_path)
