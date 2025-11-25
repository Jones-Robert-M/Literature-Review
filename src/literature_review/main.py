import click
from dotenv import load_dotenv
import os
import google.generativeai as genai
from literature_review.arxiv_fetcher import fetch_arxiv
from literature_review.semantic_scholar_fetcher import fetch_semanticscholar
from literature_review.core import (
    build_corpus,
    embed_texts,
    rank_papers,
    sentiment_analysis,
    summarize_sentiment,
    cluster_summaries,
    build_similarity_graph,
    run_girvan_newman_graph,
    visualize_graph as visualize_graph_func,
    dedupe_papers,
)
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'settings.env'))

def get_gemini_model():
    """Initializes and returns the Gemini Pro model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    return model

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

@click.group()
def cli():
    """An autonomous literature review tool."""
    if not os.getenv("GEMINI_API_KEY"):
        click.echo("Warning: GEMINI_API_KEY not found. Summarization will be disabled.")

@cli.command()
@click.argument('query')
@click.option('--max-results', default=20, help='Maximum number of papers to fetch.')
@click.option('--output-dir', default=None, help='Directory to save analysis results.')
@click.option('--visualize-graph', 'do_visualize_graph', is_flag=True, help='Generate a graph visualization of the papers.')
def arxiv(query, max_results, output_dir, do_visualize_graph):
    """Fetches papers from arXiv, analyzes them, and saves the results."""
    output_dir = create_output_dir(output_dir)
    
    click.echo(f"Fetching papers from arXiv with query: '{query}' and max results: {max_results}")
    arxiv_papers = fetch_arxiv(query, max_results=max_results)
    
    click.echo(f"Fetching papers from Semantic Scholar with query: '{query}' and max results: {max_results}")
    ss_papers = fetch_semanticscholar(query, limit=max_results)

    papers = dedupe_papers(arxiv_papers + ss_papers)
    
    if not papers:
        click.echo("No papers found.")
        return
        
    df = build_corpus(papers)

    # Embeddings
    click.echo("Embedding texts...")
    embeddings = embed_texts(df['text'].tolist())

    # Ranking
    click.echo("Ranking papers...")
    df_ranked = rank_papers(df.copy(), embeddings)
    top_10 = df_ranked.head(10)
    click.echo("\nTop 10 most significant articles:")
    for i, row in top_10.iterrows():
        click.echo(f"{i+1}. {row['title']}")

    if output_dir:
        top_10.to_csv(os.path.join(output_dir, "top_10_articles.csv"), index=False)
        click.echo(f"\nTop 10 articles saved to {os.path.join(output_dir, 'top_10_articles.csv')}")

    # Sentiment Analysis
    click.echo("\nPerforming sentiment analysis...")
    sentiment_scores = sentiment_analysis(df['abstract'].fillna("").tolist())
    df['sentiment'] = sentiment_scores
    
    sentiment_summary = summarize_sentiment(sentiment_scores)
    click.echo(sentiment_summary)

    if output_dir:
        plt.figure(figsize=(10, 6))
        df['sentiment'].hist(bins=20)
        plt.title('Sentiment Analysis of Abstracts')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Number of Papers')
        sentiment_plot_path = os.path.join(output_dir, 'sentiment_analysis.png')
        plt.savefig(sentiment_plot_path)
        click.echo(f"Sentiment analysis plot saved to {sentiment_plot_path}")
        with open(os.path.join(output_dir, "sentiment_summary.txt"), "w") as f:
            f.write(sentiment_summary)

    # Word Cloud
    click.echo("\nGenerating word cloud...")
    all_text = " ".join(df['text'].tolist())
    wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
    
    if output_dir:
        wordcloud_path = os.path.join(output_dir, 'wordcloud.png')
        wordcloud.to_file(wordcloud_path)
        click.echo(f"Word cloud saved to {wordcloud_path}")

    # Clustering
    click.echo("\nClustering papers to identify themes...")
    G_sim, sim = build_similarity_graph(embeddings, top_k=6, thresh=0.55)
    labels, communities = run_girvan_newman_graph(G_sim, max_communities=min(8, len(df)))
    df['cluster'] = df.index.map(lambda i: labels.get(i, -1))
    summaries = cluster_summaries(df, labels)
    
    if output_dir:
        with open(os.path.join(output_dir, "cluster_summaries.json"), "w") as f:
            import json
            json.dump(summaries, f, indent=2)
        click.echo(f"Cluster summaries saved to {os.path.join(output_dir, 'cluster_summaries.json')}")
        
    # Graph Visualization
    if do_visualize_graph:
        click.echo("\nGenerating graph visualization...")
        if output_dir:
            graph_path = os.path.join(output_dir, 'similarity_graph.png')
            visualize_graph_func(G_sim, labels, graph_path)
            click.echo(f"Graph visualization saved to {graph_path}")

    model = get_gemini_model()
    
    # Only summarize top 10 papers
    for i, row in top_10.iterrows():
        click.echo(f"\nTitle: {row['title']}")
        click.echo(f"Authors: {row['authors']}")
        click.echo(f"URL: {row['url']}")
        
        if model:
            prompt = f"Summarize the following abstract in 3 sentences:\n\n{row['abstract']}"
            
            try:
                response = model.generate_content(prompt)
                click.echo(f"Summary: {response.text}")
            except Exception as e:
                click.echo(f"Error summarizing paper: {e}")

if __name__ == '__main__':
    cli()