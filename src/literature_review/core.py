import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer
import Levenshtein
from .plotting_utils import set_favourite_plot_params, apply_favourite_figure_params
from itertools import combinations
from pyvis.network import Network # Import pyvis
import colorsys # For better color generation
import json # Import json for proper formatting of pyvis options
from collections import Counter # For distributions

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

# Initialize tokenizer for the summarization model to check input length
summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
MAX_SUMMARY_INPUT_LENGTH = summarization_tokenizer.model_max_length

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
    communities_generator = nx.algorithms.community.girvan_newman(G)
    communities = []
    try:
        for com in communities_generator:
            current_communities = [list(c) for c in com]
            if len(current_communities) > 1:
                communities = current_communities
            if len(communities) >= max_communities or len(G) == 0:
                break
        if not communities and len(G) > 0:
            communities = [list(G.nodes())]
    except Exception as e:
        print(f"Error during Girvan-Newman community detection: {e}")
        communities = [list(G.nodes())]

    labels = {}
    for i, com in enumerate(communities):
        for node in com:
            labels[node] = i
    return labels, communities

def cluster_summaries(df, labels, top_k_terms=15):
    summaries = {}
    for cid in sorted(set(labels.values())):
        idx = [i for i,v in labels.items() if v==cid]
        cluster_texts = df.iloc[idx]['text'].dropna().tolist()
        if not cluster_texts:
            summaries[cid] = {"n_papers": len(idx), "top_terms": [], "paper_indices": [int(i) for i in idx]}
            continue

        text = " ".join(cluster_texts)
        tf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
        
        try:
            X = tf.fit_transform([text])
            terms = np.array(tf.get_feature_names_out())
            scores = X.toarray().ravel()
            top = terms[np.argsort(-scores)][:top_k_terms].tolist()
        except ValueError:
            top = []
            
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

def _chunk_text_by_sentence(text, max_length_tokens):
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        processed_sentence = sentence.strip() + '. '
        sentence_tokens = summarization_tokenizer.encode(processed_sentence, add_special_tokens=False)

        if len(current_chunk_tokens) + len(sentence_tokens) < max_length_tokens:
            current_chunk_sentences.append(processed_sentence)
            current_chunk_tokens.extend(sentence_tokens)
        else:
            if current_chunk_sentences:
                chunks.append("".join(current_chunk_sentences).strip())
            current_chunk_sentences = [processed_sentence]
            current_chunk_tokens = sentence_tokens
    
    if current_chunk_sentences:
        chunks.append("".join(current_chunk_sentences).strip())
    
    return chunks

def summarize_text_local(text):
    """Summarizes a text using a local model, handling long texts by chunking."""
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
    tokenized_text_len = len(summarization_tokenizer.encode(text, add_special_tokens=False))
    
    if tokenized_text_len <= MAX_SUMMARY_INPUT_LENGTH:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    else:
        print(f"Warning: Text too long for summarization model ({tokenized_text_len} tokens). Chunking into smaller pieces.")
        chunks = _chunk_text_by_sentence(text, MAX_SUMMARY_INPUT_LENGTH - 50) 
        
        individual_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            try:
                chunk_summary = summarizer(chunk, max_length=80, min_length=20, do_sample=False)
                individual_summaries.append(chunk_summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk {i+1}: {e}. Skipping chunk summary.")
                individual_summaries.append(chunk) 
        
        final_summary = " ".join(individual_summaries)
        
        final_summary_token_len = len(summarization_tokenizer.encode(final_summary, add_special_tokens=False))
        if final_summary_token_len > MAX_SUMMARY_INPUT_LENGTH:
            print(f"Warning: Aggregated summary ({final_summary_token_len} tokens) is still too long. Attempting to re-summarize for conciseness.")
            final_summary = summarizer(final_summary, max_length=250, min_length=50, do_sample=False)[0]['summary_text']
        
        return final_summary


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

def get_graph_statistics(G, labels):
    """Calculates summary statistics for the graph."""
    stats = {}
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    
    if stats['num_nodes'] > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        try:
            stats['avg_clustering_coefficient'] = nx.average_clustering(G)
        except Exception:
            stats['avg_clustering_coefficient'] = 0.0 # Handle disconnected graphs

        stats['num_connected_components'] = nx.number_connected_components(G)
        stats['num_isolated_nodes'] = len(list(nx.isolates(G)))
        stats['degree_distribution'] = list(degrees) # Store for plotting
    else:
        stats['avg_degree'] = 0.0
        stats['avg_clustering_coefficient'] = 0.0
        stats['num_connected_components'] = 0
        stats['num_isolated_nodes'] = 0
        stats['degree_distribution'] = []

    # Cluster size distribution
    cluster_sizes = {}
    for cluster_id in set(labels.values()):
        cluster_sizes[cluster_id] = sum(1 for node_id, cid in labels.items() if cid == cluster_id)
    stats['cluster_size_distribution'] = list(cluster_sizes.values())
    
    return stats

def plot_degree_distribution(degree_sequence, output_path):
    """Plots the degree distribution of the graph."""
    if not degree_sequence:
        print("Warning: No degree data to plot distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(degree_sequence, bins=max(1, int(np.sqrt(len(degree_sequence)))), edgecolor='black')
    ax.set_title('Node Degree Distribution', fontsize=16, fontweight='bold')
    ax = set_favourite_plot_params(ax, x_title='Degree', y_title='Number of Nodes')
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path)
    plt.close()

def plot_cluster_size_distribution(cluster_sizes, output_path):
    """Plots the distribution of cluster sizes."""
    if not cluster_sizes:
        print("Warning: No cluster size data to plot distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cluster_sizes, bins=max(1, int(np.sqrt(len(cluster_sizes)))), edgecolor='black')
    ax.set_title('Cluster Size Distribution', fontsize=16, fontweight='bold')
    ax = set_favourite_plot_params(ax, x_title='Cluster Size (Number of Papers)', y_title='Number of Clusters')
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path)
    plt.close()

def get_keywords_distribution(df, top_n=20):
    """Extracts top N keywords and their frequencies from the corpus."""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(df['text'].dropna())
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum TF-IDF scores for each term across all documents
    sums = tfidf_matrix.sum(axis=0)
    keywords_scores = []
    for col, term in enumerate(feature_names):
        keywords_scores.append((term, sums[0, col]))
    
    keywords_scores.sort(key=lambda x: x[1], reverse=True)
    return keywords_scores[:top_n]

def plot_keywords_distribution(keywords_dist, output_path):
    """Plots the distribution of top keywords."""
    if not keywords_dist:
        print("Warning: No keyword data to plot distribution.")
        return

    keywords = [item[0] for item in keywords_dist]
    scores = [item[1] for item in keywords_dist]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(keywords, scores, color='skyblue')
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Keywords', fontsize=12, fontweight='bold')
    ax.set_title('Top Keywords Distribution', fontsize=16, fontweight='bold')
    ax.invert_yaxis() # Highest score at the top
    
    ax = set_favourite_plot_params(ax, x_title='TF-IDF Score', y_title='Keyword')
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path)
    plt.close()

def get_author_distribution(df, min_articles=5):
    """Counts articles per author and returns authors with more than min_articles."""
    all_authors = []
    for authors_str in df['authors'].dropna():
        # Handle different delimiters (e.g., ", " or ";")
        if "; " in authors_str:
            all_authors.extend([a.strip() for a in authors_str.split(';') if a.strip()])
        else:
            all_authors.extend([a.strip() for a in authors_str.split(',') if a.strip()])

    author_counts = Counter(all_authors)
    prolific_authors = {author: count for author, count in author_counts.items() if count >= min_articles}
    return sorted(prolific_authors.items(), key=lambda item: item[1], reverse=True)

def plot_author_distribution(author_dist, output_path):
    """Plots the distribution of prolific authors."""
    if not author_dist:
        print(f"Warning: No authors found with more than 5 articles to plot distribution.")
        return

    authors = [item[0] for item in author_dist]
    counts = [item[1] for item in author_dist]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(authors, counts, color='lightcoral')
    ax.set_xlabel('Number of Articles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Author', fontsize=12, fontweight='bold')
    ax.set_title('Authors with Most Articles', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    ax = set_favourite_plot_params(ax, x_title='Number of Articles', y_title='Author')
    fig = apply_favourite_figure_params(fig)
    plt.savefig(output_path)
    plt.close()

def visualize_graph(G, labels, cluster_summaries_input, output_path, df_papers_info):
    """
    Visualizes an interactive graph using Pyvis with nodes colored by cluster,
    node size based on cluster membership, and a legend for top 10 themes.
    """
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='remote')
    
    # Physics options as a Python dictionary, then dump to JSON string
    physics_options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -10000, 
          "centralGravity": 0.01,        
          "springLength": 200,           
          "springConstant": 0.03,        
          "damping": 0.09,
          "avoidOverlap": 0.9            
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "timestep": 0.5
      }
    }
    net.set_options(json.dumps(physics_options))

    # Generate a set of distinct colors for clusters
    num_unique_clusters = len(set(labels.values()))
    hsv_colors = [(i / num_unique_clusters, 0.8, 0.8) for i in range(num_unique_clusters)]
    hex_colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]]
    
    # Calculate number of edges for each cluster and enrich cluster_summaries_input
    full_cluster_info = {}
    for cid in sorted(set(labels.values())):
        cluster_nodes = [node for node, cluster_id in labels.items() if cluster_id == cid]
        n_edges = G.subgraph(cluster_nodes).number_of_edges() if cluster_nodes else 0
        
        summary = cluster_summaries_input.get(cid, {"n_papers": 0, "top_terms": [], "paper_indices": []})
        
        full_cluster_info[cid] = {
            "n_papers": summary['n_papers'],
            "top_terms": summary['top_terms'],
            "paper_indices": [int(idx) for idx in summary['paper_indices']],
            "n_edges": n_edges
        }
    
    # Sort clusters by number of papers (descending) and take the top 10
    sorted_clusters = sorted(full_cluster_info.items(), key=lambda item: item[1]['n_papers'], reverse=True)
    top_10_clusters = dict(sorted_clusters[:min(10, len(sorted_clusters))])

    # Add nodes to the network
    for node_idx, cluster_id in labels.items():
        cluster_color = hex_colors[cluster_id % len(hex_colors)]
        
        node_size = np.log1p(full_cluster_info.get(cluster_id, {}).get('n_papers', 1)) * 10 + 5 

        node_id_for_pyvis = str(node_idx)

        paper_title = df_papers_info.iloc[node_idx]['title']
        # Trim top terms for display in node title text
        top_terms_for_display = ', '.join(full_cluster_info.get(cluster_id,{}).get('top_terms',[])[:5])

        node_title_text = (
            f"Title: {paper_title}<br>"
            f"Cluster: {cluster_id}<br>"
            f"Papers in Cluster: {full_cluster_info.get(cluster_id,{}).get('n_papers',0)}<br>"
            f"Top 5 Terms: {top_terms_for_display}" # Trimmed here
        )
        
        net.add_node(node_id_for_pyvis,
                     title=node_title_text, 
                     group=cluster_id, 
                     color=cluster_color, 
                     size=node_size,
                     label=paper_title # Use actual paper title as label
                    )
    
    # Add edges to the network
    for u, v, data in G.edges(data=True):
        net.add_edge(str(u), str(v), value=data.get('weight', 0.1), color='#888888', physics=True)
    
    # Create legend HTML directly within the HTML output.
    legend_html = "<div style='position: absolute; top: 10px; right: 10px; background-color: #333; padding: 10px; border-radius: 5px; color: white; border: 1px solid #555; box-shadow: 0 0 10px rgba(0,0,0,0.5); max-height: 90%; overflow-y: auto;'>"
    legend_html += "<h4 style='color: white; margin-top: 0; border-bottom: 1px solid #555; padding-bottom: 5px;'>Top Themes:</h4>" # Changed to "Top Themes"
    for i, (cid, info) in enumerate(top_10_clusters.items()):
        color = hex_colors[cid % len(hex_colors)]
        top_terms = ", ".join(info['top_terms'][:3])
        legend_html += f"<p style='color: white; margin: 5px 0;'><span style='display:inline-block; width:15px; height:15px; background-color:{color}; border-radius:50%; margin-right:5px; border: 1px solid #555;'></span>"
        legend_html += f"C{cid} ({info['n_papers']} papers, {info['n_edges']} edges): {top_terms}</p>"
    legend_html += "</div>"
    
    # Save the interactive graph as an HTML file
    net.save_graph(output_path)
    
    # Inject the legend HTML into the generated pyvis HTML
    with open(output_path, 'r+') as f:
        html_content = f.read()
        html_content = html_content.replace('<body>', '<body>' + legend_html, 1)
        f.seek(0)
        f.truncate()
        f.write(html_content)

    return full_cluster_info # Return this for JSON output
