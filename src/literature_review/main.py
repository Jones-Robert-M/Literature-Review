import click
from dotenv import load_dotenv
import os
from literature_review.arxiv_fetcher import fetch_arxiv
from literature_review.semantic_scholar_fetcher import fetch_semanticscholar, fetch_paper_details
from literature_review.core import (
    build_corpus,
    embed_texts,
    rank_papers,
    sentiment_analysis,
    summarize_sentiment,
    cluster_summaries,
    build_similarity_graph,
    run_girvan_newman_graph,
    visualize_graph,
    summarize_text_local,
    dedupe_papers,
    calculate_reference_overlap,
    plot_reference_overlap,
)
from literature_review.plotting_utils import set_favourite_plot_params, apply_favourite_figure_params
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import distinctipy
import requests # Import requests to catch its exceptions
import json

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

@click.group()
def cli():
    """An autonomous literature review tool."""
    pass

@cli.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of papers to fetch.')
@click.option('--output-dir', default="output", help='Directory to save analysis results. Defaults to "output".')
@click.option('--visualize-graph/--no-visualize-graph', 'do_visualize_graph', default=True, help='Generate a graph visualization of the papers.')
@click.option('--sub-search/--no-sub-search', default=True, help='Perform a sub-search on referenced articles.')
def arxiv(query, max_results, output_dir, do_visualize_graph, sub_search):
    """Fetches papers from arXiv, analyzes them, and saves the results."""
    output_dir = create_output_dir(output_dir)
    
    arxiv_papers = []
    try:
        click.echo(f"Fetching papers from arXiv with query: '{query}' and max results: {max_results}")
        arxiv_papers = fetch_arxiv(query, max_results=max_results)
    except requests.exceptions.ConnectionError as e:
        click.echo(f"Warning: Could not fetch papers from arXiv. Reason: {e}")
        click.echo("This might be a temporary network issue or DNS problem. Continuing without arXiv papers.")
    except Exception as e:
        click.echo(f"An unexpected error occurred while fetching from arXiv: {e}")
        click.echo("Continuing without arXiv papers.")


    ss_papers = []
    try:
        click.echo(f"Fetching papers from Semantic Scholar with query: '{query}' and max results: {max_results}")
        ss_papers = fetch_semanticscholar(query, limit=max_results)
    except Exception as e:
        click.echo(f"Warning: Could not fetch papers from Semantic Scholar. Reason: {e}")
        click.echo("Continuing with arXiv papers only (if any were fetched).")

    initial_papers = dedupe_papers(arxiv_papers + ss_papers)
    papers = initial_papers.copy()

    if sub_search:
        click.echo("\nPerforming sub-search on referenced articles...")
        referenced_ids = set()
        for paper in initial_papers:
            if 'references' in paper:
                referenced_ids.update(paper['references'])
        
        click.echo(f"Found {len(referenced_ids)} unique referenced articles.")
        
        if referenced_ids:
            try:
                referenced_papers = fetch_paper_details(list(referenced_ids))
                click.echo(f"Fetched details for {len(referenced_papers)} referenced articles.")
                
                initial_paper_ids = {p['id'] for p in initial_papers if 'id' in p}
                new_papers = [p for p in referenced_papers if 'id' in p and p['id'] not in initial_paper_ids]
                
                click.echo(f"Found {len(new_papers)} new articles from sub-search.")
                
                papers.extend(new_papers)
                papers = dedupe_papers(papers)

                overlaps = calculate_reference_overlap(initial_papers)
                if overlaps and output_dir:
                    plot_reference_overlap(overlaps, os.path.join(output_dir, "reference_overlap.png"))
                    click.echo(f"Reference overlap plot saved to {os.path.join(output_dir, 'reference_overlap.png')}")
            except Exception as e:
                click.echo(f"Warning: Could not perform sub-search. Reason: {e}")
                click.echo("This is likely due to a missing or invalid Semantic Scholar API key.")


    if not papers:
        click.echo("No papers found from any source. Exiting.")
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
        fig, ax = plt.subplots(figsize=(10, 6))
        df['sentiment'].hist(bins=20, ax=ax)
        ax.set_title('Sentiment Analysis of Abstracts', fontsize=16, fontweight='bold')
        ax = set_favourite_plot_params(ax, x_title='Sentiment Score', y_title='Number of Papers')
        fig = apply_favourite_figure_params(fig)
        sentiment_plot_path = os.path.join(output_dir, 'sentiment_analysis.png')
        plt.savefig(sentiment_plot_path)
        plt.close() # Close figure to free memory
        click.echo(f"Sentiment analysis plot saved to {sentiment_plot_path}")
        with open(os.path.join(output_dir, "sentiment_summary.txt"), "w") as f:
            f.write(sentiment_summary)

    # Word Cloud
    click.echo("\nGenerating word cloud...")
    all_text = " ".join(df['text'].tolist())
    
    if all_text:
        # Generate distinct colors
        num_unique_words = len(set(word.lower() for word in all_text.split() if word.isalpha())) # Count actual unique words
        # Limit colors to a reasonable number to prevent performance issues and ensure visual distinction
        colors_to_generate = min(num_unique_words, 100) 
        if colors_to_generate > 0:
            colors = distinctipy.get_colors(colors_to_generate)
        else: # Fallback if no words to color
            colors = [(0.0, 0.0, 0.0)] # Black
        
        # Create a color function for the word cloud
        color_idx = 0
        def get_distinct_color(word, font_size, position, orientation, random_state=None, **kwargs):
            nonlocal color_idx
            color = colors[color_idx % len(colors)]
            color_idx += 1
            return distinctipy.get_hex(color)

        wordcloud = WordCloud(width=1600, height=800, background_color="white", color_func=get_distinct_color).generate(all_text)
        
        if output_dir:
            wordcloud_path = os.path.join(output_dir, 'wordcloud.png')
            wordcloud.to_file(wordcloud_path)
            click.echo(f"Word cloud saved to {wordcloud_path}")
    else:
        click.echo("No text available to generate word cloud.")

    # Clustering
    click.echo("\nClustering papers to identify themes...")
    G_sim, sim = build_similarity_graph(embeddings, top_k=6, thresh=0.55)
    labels, communities = run_girvan_newman_graph(G_sim, max_communities=min(8, len(df)))
    df['cluster'] = df.index.map(lambda i: labels.get(i, -1))
    summaries = cluster_summaries(df, labels)
    
    # Graph Visualization
    full_cluster_info = {}
    if do_visualize_graph:
        click.echo("\nGenerating graph visualization...")
        if output_dir:
            graph_path = os.path.join(output_dir, 'similarity_graph.html') # Changed to HTML
            full_cluster_info = visualize_graph(G_sim, labels, summaries, graph_path, df) # Pass df here
            click.echo(f"Graph visualization saved to {graph_path}")

    # Save cluster info including edge counts
    if output_dir and full_cluster_info:
        cluster_info_path = os.path.join(output_dir, "graph_cluster_info.json")
        with open(cluster_info_path, "w") as f:
            json.dump(full_cluster_info, f, indent=2)
        click.echo(f"Graph cluster information saved to {cluster_info_path}")
        
    if output_dir:
        with open(os.path.join(output_dir, "cluster_summaries.json"), "w") as f:
            json.dump(summaries, f, indent=2)
        click.echo(f"Cluster summaries saved to {os.path.join(output_dir, 'cluster_summaries.json')}")

    # Theme-specific Summaries
    click.echo("\nSummarizing top 10 themes...")
    if full_cluster_info:
        # Sort clusters by number of papers (descending) and take the top 10
        sorted_clusters = sorted(full_cluster_info.items(), key=lambda item: item[1]['n_papers'], reverse=True)
        top_10_clusters = dict(sorted_clusters[:min(10, len(sorted_clusters))])

        for cid, info in top_10_clusters.items():
            click.echo(f"\n--- Theme {cid} (Top Terms: {', '.join(info['top_terms'])}) ---")
            theme_abstracts = " ".join(df.iloc[info['paper_indices']]['abstract'].fillna("").tolist())
            if len(theme_abstracts.strip()) > 0:
                try:
                    theme_summary = summarize_text_local(theme_abstracts)
                    click.echo(f"Summary: {theme_summary}")
                    if output_dir:
                        theme_summary_path = os.path.join(output_dir, f"theme_summary_C{cid}.txt")
                        with open(theme_summary_path, "w") as f:
                            f.write(theme_summary)
                        click.echo(f"Theme summary saved to {theme_summary_path}")
                except Exception as e:
                    click.echo(f"Error summarizing theme {cid}: {e}")
            else:
                click.echo("No abstracts available for this theme to summarize.")
    else:
        click.echo("No cluster information available to summarize themes.")


    # Summarization of top 10 ranked papers (original functionality)
    click.echo("\nSummarizing top 10 ranked papers...")
    for i, row in top_10.iterrows():
        click.echo(f"\nTitle: {row['title']}")
        click.echo(f"Authors: {row['authors']}")
        click.echo(f"URL: {row['url']}")
        
        try:
            summary = summarize_text_local(row['abstract'])
            click.echo(f"Summary: {summary}")
        except Exception as e:
            click.echo(f"Error summarizing paper: {e}")

if __name__ == '__main__':
    cli()
