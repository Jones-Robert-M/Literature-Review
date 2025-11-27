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
    get_graph_statistics,
    plot_degree_distribution,
    plot_cluster_size_distribution,
    get_keywords_distribution,
    plot_keywords_distribution,
    get_author_distribution,
    plot_author_distribution,
)
from literature_review.plotting_utils import set_favourite_plot_params, apply_favourite_figure_params
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import distinctipy
import requests # Import requests to catch its exceptions
import json
import openai # Import openai

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'settings.env'))

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_openai_client():
    """Initializes and returns the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)

def generate_analysis_report(openai_client, analysis_results, query_name, output_path):
    """Generates a comprehensive analysis report using OpenAI."""
    if not openai_client:
        click.echo("Warning: OpenAI API key not found. Skipping report generation.")
        return

    prompt = (
        f"Generate a comprehensive and insightful report based on the following literature review analysis for the query: '{query_name}'.\n\n"
        f"**Graph Summary Statistics:**\n{json.dumps(analysis_results.get('graph_stats_summary', {}), indent=2)}\n\n" # Ensure JSON string
        f"**Sentiment Analysis Summary:**\n{analysis_results.get('sentiment_summary', 'N/A')}\n\n"
        f"**Top 10 Papers:**\n"
    )
    for i, paper in enumerate(analysis_results.get('top_10_papers', [])):
        prompt += f"{i+1}. {paper['title']} (Authors: {paper['authors']}, URL: {paper['url']})\n"
    
    prompt += f"\n**Top 10 Themes (Clusters):**\n"
    for cid, info in analysis_results.get('top_10_themes', {}).items():
        prompt += f"Cluster {cid} ({info['n_papers']} papers, {info['n_edges']} edges): Top Terms: {', '.join(info['top_terms'])}\n"
    
    prompt += "\n**Objective:** Provide a well-structured report that includes an introduction, a summary of key findings (e.g., overarching themes, significant papers, sentiment trends, network characteristics), a discussion of potential research gaps or promising future directions, and a conclusion. Make it concise yet informative, suitable for a researcher. Focus on synthesizing the information to derive insights, rather than just listing data points. The report should be around 500-800 words."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4" if available and desired
            messages=[
                {"role": "system", "content": "You are a helpful research assistant specialized in synthesizing literature review analysis results into concise reports."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500, # Adjust as needed
            temperature=0.7,
        )
        report_content = response.choices[0].message.content
        with open(output_path, "w") as f:
            f.write(report_content)
        click.echo(f"Analysis report generated and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error generating analysis report with OpenAI: {e}")


def _perform_core_analysis(query_name, papers_list, output_base_dir, do_visualize_graph, do_sub_search, do_generate_report):
    """
    Performs the core analysis steps (embeddings, ranking, sentiment, word cloud, clustering, graph visualization, summaries)
    on a given list of papers and saves results to output_base_dir.
    """
    if not papers_list:
        click.echo(f"No papers found for '{query_name}'. Skipping core analysis.")
        return

    df = build_corpus(papers_list)

    # Save all paper details
    if output_base_dir:
        # Define a list of columns that should be saved, checking for their existence in df
        cols_to_save = ['title', 'abstract', 'authors', 'published', 'source', 'doi', 'url', 'citationCount']
        existing_cols = [col for col in cols_to_save if col in df.columns]
        
        all_papers_details_path = os.path.join(output_base_dir, "all_papers_details.csv")
        df[existing_cols].to_csv(all_papers_details_path, index=False)
        click.echo(f"All paper details saved to {all_papers_details_path}")

    # Embeddings
    click.echo("Embedding texts...")
    embeddings = embed_texts(df['text'].tolist())

    # Ranking
    click.echo("Ranking papers...")
    df_ranked = rank_papers(df.copy(), embeddings)
    top_10 = df_ranked.head(10)
    click.echo(f"\nTop 10 most significant articles for '{query_name}':")
    for i, row in top_10.iterrows():
        click.echo(f"{i+1}. {row['title']}")

    if output_base_dir:
        top_10.to_csv(os.path.join(output_base_dir, "top_10_articles.csv"), index=False)
        click.echo(f"\nTop 10 articles saved to {os.path.join(output_base_dir, 'top_10_articles.csv')}")

    # Sentiment Analysis
    click.echo("\nPerforming sentiment analysis...")
    sentiment_scores = sentiment_analysis(df['abstract'].fillna("").tolist())
    df['sentiment'] = sentiment_scores
    
    sentiment_summary = summarize_sentiment(sentiment_scores)
    click.echo(sentiment_summary)

    if output_base_dir:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['sentiment'].hist(bins=20, ax=ax)
        ax.set_title(f'Sentiment Analysis of Abstracts for {query_name}', fontsize=16, fontweight='bold')
        ax = set_favourite_plot_params(ax, x_title='Sentiment Score', y_title='Number of Papers')
        fig = apply_favourite_figure_params(fig)
        sentiment_plot_path = os.path.join(output_base_dir, 'sentiment_analysis.png')
        plt.savefig(sentiment_plot_path)
        plt.close() # Close figure to free memory
        click.echo(f"Sentiment analysis plot saved to {sentiment_plot_path}")
        with open(os.path.join(output_base_dir, "sentiment_summary.txt"), "w") as f:
            f.write(sentiment_summary)

    # Word Cloud
    click.echo("\nGenerating word cloud...")
    all_text = " ".join(df['text'].tolist())

    # Filter out query words from the word cloud text
    query_words = {word.lower() for word in query_name.split()}
    filtered_text = " ".join([word for word in all_text.split() if word.lower() not in query_words])
    
    if filtered_text.strip():
        num_unique_words = len(set(word.lower() for word in filtered_text.split() if word.isalpha()))
        colors_to_generate = min(num_unique_words, 100) 
        if colors_to_generate > 0:
            colors = distinctipy.get_colors(colors_to_generate)
        else:
            colors = [(0.0, 0.0, 0.0)]
        
        color_idx = 0
        def get_distinct_color(word, font_size, position, orientation, random_state=None, **kwargs):
            nonlocal color_idx
            color = colors[color_idx % len(colors)]
            color_idx += 1
            return distinctipy.get_hex(color)

        wordcloud = WordCloud(width=1600, height=800, background_color="white", color_func=get_distinct_color).generate(filtered_text)
        
        if output_base_dir:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(query_name, fontsize=20, fontweight='bold', pad=20)
            ax.axis("off")
            
            wordcloud_path = os.path.join(output_base_dir, 'wordcloud.png')
            fig.savefig(wordcloud_path, bbox_inches='tight')
            plt.close(fig) # Close figure to free memory
            click.echo(f"Word cloud saved to {wordcloud_path}")
    else:
        click.echo("No text available to generate word cloud after filtering query words.")

    # Clustering
    click.echo("\nClustering papers to identify themes...")
    G_sim, sim = build_similarity_graph(embeddings, top_k=6, thresh=0.55)
    labels, communities = run_girvan_newman_graph(G_sim, max_communities=min(8, len(df)))
    df['cluster'] = df.index.map(lambda i: labels.get(i, -1))
    summaries = cluster_summaries(df, labels)
    
    # Graph Visualization
    full_cluster_info = {} # Corrected initialization
    if do_visualize_graph:
        click.echo("\nGenerating graph visualization...")
        if output_base_dir:
            graph_path = os.path.join(output_base_dir, 'similarity_graph.html') 
            full_cluster_info = visualize_graph(G_sim, labels, summaries, graph_path, df) # Pass df here
            click.echo(f"Graph visualization saved to {graph_path}")

    # Graph Statistics
    click.echo("\nCalculating graph summary statistics...")
    graph_stats = get_graph_statistics(G_sim, labels)
    
    click.echo("\n--- Graph Summary Statistics ---")
    for key, value in graph_stats.items():
        if key not in ['degree_distribution', 'cluster_size_distribution']:
            click.echo(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    if output_base_dir:
        stats_df = pd.DataFrame([graph_stats]).drop(columns=['degree_distribution', 'cluster_size_distribution'], errors='ignore')
        stats_path = os.path.join(output_base_dir, "graph_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        click.echo(f"Graph summary statistics saved to {stats_path}")

        plot_degree_distribution(graph_stats['degree_distribution'], os.path.join(output_base_dir, "degree_distribution.png") )
        click.echo(f"Degree distribution plot saved to {os.path.join(output_base_dir, 'degree_distribution.png')}")

        plot_cluster_size_distribution(graph_stats['cluster_size_distribution'], os.path.join(output_base_dir, "cluster_size_distribution.png") )
        click.echo(f"Cluster size distribution plot saved to {os.path.join(output_base_dir, 'cluster_size_distribution.png')}")

    # Keywords distribution
    click.echo("\nGenerating keyword distribution plot...")
    keywords_dist = get_keywords_distribution(df)
    if keywords_dist and output_base_dir:
        plot_keywords_distribution(keywords_dist, os.path.join(output_base_dir, "keywords_distribution.png") )
        click.echo(f"Keyword distribution plot saved to {os.path.join(output_base_dir, 'keywords_distribution.png')}")

    # Author distribution
    click.echo("\nGenerating author distribution plot...")
    author_dist = get_author_distribution(df, min_articles=5)
    if author_dist and output_base_dir:
        plot_author_distribution(author_dist, os.path.join(output_base_dir, "author_distribution.png") )
        click.echo(f"Author distribution plot saved to {os.path.join(output_base_dir, 'author_distribution.png')}")


    # Save cluster info including edge counts
    if output_base_dir and full_cluster_info:
        cluster_info_path = os.path.join(output_base_dir, "graph_cluster_info.json")
        with open(cluster_info_path, "w") as f:
            json.dump(full_cluster_info, f, indent=2)
        click.echo(f"Graph cluster information saved to {cluster_info_path}")
        
    if output_base_dir:
        with open(os.path.join(output_base_dir, "cluster_summaries.json"), "w") as f:
            json.dump(summaries, f, indent=2)
        click.echo(f"Cluster summaries saved to {os.path.join(output_base_dir, 'cluster_summaries.json')}")

    # Theme-specific Summaries
    click.echo("\nSummarizing top 10 themes...")
    top_10_themes_for_report = {} # Corrected initialization
    if full_cluster_info:
        sorted_clusters = sorted(full_cluster_info.items(), key=lambda item: item[1]['n_papers'], reverse=True)
        top_10_clusters = dict(sorted_clusters[:min(10, len(sorted_clusters))])
        top_10_themes_for_report = top_10_clusters # Store for report

        for cid, info in top_10_clusters.items():
            click.echo(f"\n--- Theme {cid} (Top Terms: {', '.join(info['top_terms'])}) ---")
            theme_abstracts = " ".join(df.iloc[info['paper_indices']]['abstract'].fillna("").tolist())
            if len(theme_abstracts.strip()) > 0:
                try:
                    theme_summary = summarize_text_local(theme_abstracts)
                    click.echo(f"Summary: {theme_summary}")
                    if output_base_dir:
                        theme_summary_path = os.path.join(output_base_dir, f"theme_summary_C{cid}.txt")
                        with open(theme_summary_path, "w") as f:
                            f.write(theme_summary)
                        click.echo(f"Theme summary saved to {theme_summary_path}")
                except Exception as e:
                    click.echo(f"Error summarizing theme {cid}: {e}")
            else:
                click.echo("No abstracts available for this theme to summarize.")
    else:
        click.echo("No cluster information available to summarize themes.")

    # Summarization of top 10 ranked papers
    click.echo("\nSummarizing top 10 ranked papers...")
    top_10_papers_for_report = []
    for i, row in top_10.iterrows():
        click.echo(f"\nTitle: {row['title']}")
        click.echo(f"Authors: {row['authors']}")
        click.echo(f"URL: {row['url']}")
        
        try:
            summary = summarize_text_local(row['abstract'])
            click.echo(f"Summary: {summary}")
            top_10_papers_for_report.append({"title": row['title'], "authors": row['authors'], "url": row['url'], "summary": summary})
        except Exception as e:
            click.echo(f"Error summarizing paper: {e}")
            top_10_papers_for_report.append({"title": row['title'], "authors": row['authors'], "url": row['url'], "summary": f"Error: {e}"})

    # Generate full analysis report
    if do_generate_report:
        click.echo("\nGenerating full analysis report...")
        openai_client = get_openai_client()
        if openai_client:
            analysis_results = {
                "graph_stats_summary": graph_stats,
                "sentiment_summary": sentiment_summary,
                "top_10_papers": top_10_papers_for_report,
                "top_10_themes": top_10_themes_for_report
            }
            report_path = os.path.join(output_base_dir, "analysis_report.txt")
            generate_analysis_report(openai_client, analysis_results, query_name, report_path)


    click.echo(f"--- Finished analysis for: '{query_name}' ---")


@click.group()
def cli():
    """An autonomous literature review tool."""
    pass

@cli.command(name="single-arxiv")
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of papers to fetch.')
@click.option('--output-dir', default="output", help='Directory to save analysis results. Defaults to "output".')
@click.option('--visualize-graph/--no-visualize-graph', 'do_visualize_graph', default=True, help='Generate a graph visualization of the papers.')
@click.option('--sub-search/--no-sub-search', 'do_sub_search', default=True, help='Perform a sub-search on referenced articles.')
@click.option('--generate-report/--no-generate-report', 'do_generate_report', default=True, help='Generate a full analysis report using OpenAI.')
def single_arxiv_command(query, max_results, output_dir, do_visualize_graph, do_sub_search, do_generate_report):
    """Fetches papers from arXiv, analyzes them, and saves the results for a single query."""
    papers_for_analysis = []
    # Fetch initial papers for this single query
    try:
        click.echo(f"Fetching papers from arXiv with query: '{query}' and max results: {max_results}")
        arxiv_res = fetch_arxiv(query, max_results=max_results)
        papers_for_analysis.extend(arxiv_res)
    except requests.exceptions.ConnectionError as e:
        click.echo(f"Warning: Could not fetch papers from arXiv for query '{query}'. Reason: {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred while fetching from arXiv for query '{query}': {e}")

    try:
        click.echo(f"Fetching papers from Semantic Scholar with query: '{query}' and max results: {max_results}")
        ss_res = fetch_semanticscholar(query, limit=max_results)
        papers_for_analysis.extend(ss_res)
    except Exception as e:
        click.echo(f"Warning: Could not fetch papers from Semantic Scholar for query '{query}'. Reason: {e}")
        click.echo("This may be due to a missing or invalid Semantic Scholar API key.")

    # Deduplicate before passing to analysis pipeline
    initial_deduplicated_papers = dedupe_papers(papers_for_analysis)

    # Determine query-specific output directory
    query_output_subdir = os.path.join(output_dir, query.replace(" ", "_").replace("/", "_").replace("\\\\", "_").lower()) 
    create_output_dir(query_output_subdir) # Ensure it exists

    # Run the analysis pipeline for this single query
    _perform_core_analysis(query, initial_deduplicated_papers, query_output_subdir, do_visualize_graph, do_sub_search, do_generate_report)
    
# Alias for single-arxiv
cli.add_command(single_arxiv_command, name="arxiv")

@cli.command()
@click.argument('queries', nargs=-1, required=True)
@click.option('--max-results', default=10, help='Maximum number of papers to fetch per query (for initial fetch).')
@click.option('--output-dir', default="output", help='Main directory to save all aggregated analysis results. Defaults to "output".')
@click.option('--visualize-graph/--no-visualize-graph', 'do_visualize_graph', default=True, help='Generate a single aggregated graph visualization of the papers.')
@click.option('--sub-search/--no-sub-search', 'do_sub_search', default=True, help='Perform a sub-search on referenced articles from the aggregated initial papers.')
@click.option('--generate-report/--no-generate-report', 'do_generate_report', default=True, help='Generate a full analysis report using OpenAI.')
def multi_arxiv(queries, max_results, output_dir, do_visualize_graph, do_sub_search, do_generate_report):
    """Fetches papers from arXiv for multiple queries, aggregates them, analyzes them, and saves the results."""
    click.echo(f"--- Starting aggregated analysis for {len(queries)} queries ---")
    
    main_output_dir = create_output_dir(output_dir) # Ensure base output dir exists

    all_fetched_papers = []

    # Fetch papers for all queries
    for query in queries:
        click.echo(f"\nFetching papers for query: '{query}'...")
        current_query_papers = []
        # Arxiv
        try:
            arxiv_res = fetch_arxiv(query, max_results=max_results)
            current_query_papers.extend(arxiv_res)
        except requests.exceptions.ConnectionError as e:
            click.echo(f"Warning: Could not fetch papers from arXiv for query '{query}'. Reason: {e}")
        except Exception as e:
            click.echo(f"An unexpected error occurred while fetching from arXiv for query '{query}': {e}")

        # Semantic Scholar
        try:
            ss_res = fetch_semanticscholar(query, limit=max_results)
            current_query_papers.extend(ss_res)
        except Exception as e:
            click.echo(f"Warning: Could not fetch papers from Semantic Scholar for query '{query}'. Reason: {e}")
            click.echo("This may be due to a missing or invalid Semantic Scholar API key.")
        
        all_fetched_papers.extend(current_query_papers) # Add papers from current query to overall list

    # Combine and deduplicate all fetched papers from all queries
    aggregated_papers_deduplicated = dedupe_papers(all_fetched_papers)

    # Now run the core analysis pipeline on the aggregated, deduplicated papers
    _perform_core_analysis(
        query_name="Aggregated Corpus", # Use a generic name for aggregated output
        papers_list=aggregated_papers_deduplicated,
        output_base_dir=main_output_dir,
        do_visualize_graph=do_visualize_graph,
        do_sub_search=do_sub_search,
        do_generate_report=do_generate_report
    )

    click.echo("\n--- Finished aggregated analysis ---")

if __name__ == '__main__':
    cli()