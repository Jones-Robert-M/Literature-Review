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
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    
    click.echo(f"Fetching papers from arXiv with query: '{query}' and max results: {max_results}")
    arxiv_papers = fetch_arxiv(query, max_results=max_results)
    
    ss_papers = []
    try:
        click.echo(f"Fetching papers from Semantic Scholar with query: '{query}' and max results: {max_results}")
        ss_papers = fetch_semanticscholar(query, limit=max_results)
    except Exception as e:
        click.echo(f"Warning: Could not fetch papers from Semantic Scholar. Reason: {e}")
        click.echo("Continuing with arXiv papers only.")

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
        fig, ax = plt.subplots(figsize=(10, 6))
        df['sentiment'].hist(bins=20, ax=ax)
        ax.set_title('Sentiment Analysis of Abstracts', fontsize=16, fontweight='bold')
        ax = set_favourite_plot_params(ax, x_title='Sentiment Score', y_title='Number of Papers')
        fig = apply_favourite_figure_params(fig)
        sentiment_plot_path = os.path.join(output_dir, 'sentiment_analysis.png')
        plt.savefig(sentiment_plot_path)
        plt.close()
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
            visualize_graph(G_sim, labels, summaries, graph_path)
            click.echo(f"Graph visualization saved to {graph_path}")

    # Corpus Summarization
    click.echo("\nSummarizing the entire corpus...")
    all_abstracts = " ".join(df['abstract'].fillna("").tolist())
    if len(all_abstracts.strip()) > 0:
        corpus_summary = summarize_text_local(all_abstracts)
        click.echo("\nCorpus Summary:")
        click.echo(corpus_summary)
        if output_dir:
            with open(os.path.join(output_dir, "corpus_summary.txt"), "w") as f:
                f.write(corpus_summary)
            click.echo(f"\nCorpus summary saved to {os.path.join(output_dir, 'corpus_summary.txt')}")

    # Summarization of top 10
    click.echo("\nSummarizing top 10 papers...")
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
