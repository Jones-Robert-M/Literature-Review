# Literature Review

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

An autonomous literature review tool that uses large language models to help you conduct literature reviews efficiently.

## Features

-   **Multi-Query Processing**: Analyze multiple research topics in a single run, with an option for aggregated analysis across all queries.
-   **Fetch Papers**: Gathers papers from arXiv and Semantic Scholar. Robust error handling ensures the process continues even if one source is temporarily unavailable.
-   **First-Order Sub-Search**: Automatically performs a sub-search on articles referenced by the initial query. *Requires a Semantic Scholar API Key for full functionality* to effectively retrieve reference data.
-   **In-depth Analysis**: Comprehensive analysis including ranking of papers, sentiment analysis of abstracts, and thematic clustering.
-   **Theme-Specific Summaries**: Instead of a single, potentially diluted corpus summary, the tool identifies the top 10 most significant themes (clusters) and provides a concise summary for each, offering focused insights.
-   **Enhanced Visualizations**:
    *   **Word Clouds**: Generates visually distinct word clouds using `distinctipy` for a clear representation of key terms.
    *   **Interactive Similarity Graphs**: Creates interactive HTML network graphs (`similarity_graph.html`) where nodes represent papers, sized logarithmically by cluster membership, and positioned with adjusted physics for better visual separation. Node labels show the paper title, and hover information provides paper title, cluster ID, number of papers in cluster, and up to 5 top terms for the cluster. The graph includes a rich legend highlighting the top 10 themes with their paper and edge counts.
-   **Graph Summary Statistics**: Outputs key statistics about the generated similarity network to the console and a CSV file (`graph_statistics.csv`). Also generates plots for degree distribution (`degree_distribution.png`) and cluster size distribution (`cluster_size_distribution.png`).
-   **Keyword and Author Distribution**: Generates bar charts showing the distribution of top keywords (`keywords_distribution.png`) and authors with more than 5 articles (`author_distribution.png`).
-   **Full Analysis Report (OpenAI Powered)**: Generates a comprehensive text document summarizing the analysis findings and suggesting future research directions (optional, requires OpenAI API Key).
-   **All Paper Details Output**: Saves a CSV file (`all_papers_details.csv`) containing titles, abstracts, authors, and other metadata for all processed papers.
-   **Local Summarization**: Utilizes a local `transformers` model for efficient summarization of abstracts and theme-specific content, gracefully handling large text inputs through intelligent chunking.
-   **Extensible**: Designed with modularity, allowing for easy expansion to support additional data sources and advanced analysis techniques.

## Getting Started

### Prerequisites

-   Python 3.8 or higher
-   pip

### Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd literature-review
    ```
3.  Install the project in editable mode, which also installs all necessary dependencies:
    ```bash
    pip install -e .
    ```

### Configuration

This tool can leverage external APIs for enhanced functionality. You need to configure your API keys.

1.  Create a copy of the `settings.env.example` file in the `configs` directory and name it `settings.env`:
    ```bash
    cp configs/settings.env.example configs/settings.env
    ```
2.  Add your API keys to the `configs/settings.env` file.
    ```
    SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-api-key"
    OPENAI_API_KEY="your-openai-api-key"
    ```
    *You can obtain a Semantic Scholar API key from [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api).*
    *You can obtain an OpenAI API key from [https://platform.openai.com/](https://platform.openai.com/).*

## Usage

The tool provides two main commands: `arxiv` (alias for `single-arxiv`) for single queries and `multi-arxiv` for batch processing and aggregated analysis.

### Single Query Analysis (`arxiv` or `single-arxiv`)

To fetch and analyze papers for a single query:

```bash
literature-review arxiv "your query" [OPTIONS]
```

**Options (all enabled by default unless specified otherwise):**

-   `--max-results`: Maximum number of initial papers to fetch (default: 10).
-   `--output-dir`: Main directory to save analysis results (default: `output`). A subdirectory named after the query will be created within this.
-   `--visualize-graph/--no-visualize-graph`: Generate an interactive HTML graph visualization of the papers (default: enabled).
-   `--sub-search/--no-sub-search`: Perform a first-order sub-search on referenced articles (default: enabled). *Requires `SEMANTIC_SCHOLAR_API_KEY` for full functionality.*
-   `--generate-report/--no-generate-report`: Generate a full analysis report using OpenAI (default: enabled). *Requires `OPENAI_API_KEY`.*

**Example:**

```bash
literature-review arxiv "deep learning architectures" --max-results 20
```
This command will fetch 20 papers, perform all analyses, generate an interactive graph and theme summaries, saving all outputs into `output/deep_learning_architectures/`.

### Multiple Query Analysis (`multi-arxiv`)

To process several research topics and aggregate their results into a single comprehensive analysis:

```bash
literature-review multi-arxiv "query 1" "query 2" "query N" [OPTIONS]
```

**Options (all enabled by default unless specified otherwise):**

-   `queries`: Space-separated list of research queries.
-   `--max-results`: Maximum number of papers to fetch *per query* (default: 10).
-   `--output-dir`: Main directory to save all *aggregated* analysis results (default: `output`). Outputs will be saved directly into this directory, not in query-specific subdirectories.
-   `--visualize-graph/--no-visualize-graph`: Generate a single aggregated interactive HTML graph visualization of the papers (default: enabled).
-   `--sub-search/--no-sub-search`: Perform a sub-search on referenced articles from the *aggregated* initial papers (default: enabled). *Requires `SEMANTIC_SCHOLAR_API_KEY` for full functionality.*
-   `--generate-report/--no-generate-report`: Generate a full analysis report using OpenAI for the aggregated corpus (default: enabled). *Requires `OPENAI_API_KEY`.*

**Example:**

```bash
literature-review multi-arxiv "large language models" "reinforcement learning in robotics" --max-results 15 --output-dir batch_analysis --no-sub-search
```
This will run analyses for two queries, aggregating their results, and saving all outputs into `batch_analysis/`. No sub-search will be performed.

## Outputs

All output files for a given query (or aggregated analysis) are saved in a dedicated directory (e.g., `output/your_query_slug/` for `arxiv` command, or `output/` for `multi-arxiv` command's aggregated output).

-   `all_papers_details.csv`: A CSV file containing titles, abstracts, authors, and other metadata for all processed papers.
-   `top_10_articles.csv`: A CSV file with the top 10 most relevant papers.
-   `sentiment_analysis.png`: A histogram of the sentiment scores of the paper abstracts.
-   `sentiment_summary.txt`: A text file with a summary of the sentiment analysis.
-   `wordcloud.png`: A word cloud generated from the paper titles and abstracts, with distinct colors for each word.
-   `cluster_summaries.json`: A JSON file with a summary of each paper cluster, including paper counts, top terms, and paper indices.
-   `graph_cluster_info.json`: A JSON file containing detailed information about each cluster in the similarity graph, including paper counts, top terms, and edge counts.
-   `similarity_graph.html`: An interactive network graph visualization.
-   `degree_distribution.png`: A plot showing the distribution of node degrees in the similarity graph.
-   `cluster_size_distribution.png`: A plot showing the distribution of cluster sizes in the similarity graph.
-   `keywords_distribution.png`: A plot showing the distribution of top keywords across the corpus.
-   `author_distribution.png`: A plot showing the distribution of authors who have published more than 5 articles in the corpus.
-   `theme_summary_C<id>.txt`: A dedicated text file for each of the top 10 themes, containing a summarized overview of the theme.
-   `reference_overlap.png`: (Only if `sub-search` is enabled and Semantic Scholar API key is functional) A histogram showing the distribution of shared references between articles.
-   `graph_statistics.csv`: A CSV file summarizing key statistics of the generated network graph (e.g., number of nodes/edges, average degree, clustering coefficient).
-   `analysis_report.txt`: (Only if `generate-report` is enabled and OpenAI API key is functional) A comprehensive text report generated by OpenAI summarizing the analysis.

## Project Structure

```
.
├── configs/
│   ├── settings.env
│   └── settings.env.example
├── src/
│   └── literature_review/
│       ├── __init__.py
│       ├── arxiv_fetcher.py
│       ├── core.py
│       ├── main.py
│       ├── plotting_utils.py
│       └── semantic_scholar_fetcher.py
├── output/
├── explanation.txt
├── LICENSE
├── pyproject.toml
└── README.md
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.
