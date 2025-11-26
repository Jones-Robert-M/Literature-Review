# Literature Review

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

An autonomous literature review tool that uses large language models to help you conduct literature reviews.

## Features

-   **Fetch Papers**: Fetches papers from arXiv and Semantic Scholar.
-   **First-Order Sub-Search**: Automatically performs a sub-search on articles referenced by the initial query (requires Semantic Scholar API Key for full functionality).
-   **In-depth Analysis**: Performs sentiment analysis, clustering, and ranking of papers.
-   **Corpus Summary**: Generates a textual summary of the entire aggregated corpus of abstracts.
-   **Visualizations**: Generates word clouds (with distinct colors for words) and network graphs to visualize the relationships between papers. The network graph highlights the 10 largest themes (clusters) and includes the number of edges attached to each theme.
-   **Detailed Cluster Information**: Outputs a JSON file containing comprehensive information about each cluster, including paper counts, top terms, and edge counts.
-   **Local Summarization**: Uses a local summarization model to summarize the abstracts of the top papers and the entire corpus.
-   **Extensible**: Easily extensible to support other data sources and analysis methods.

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
3.  Install the project in editable mode:
    ```bash
    pip install -e .
    ```

### Configuration

This tool can use the Semantic Scholar API for sub-searches and the Gemini API to summarize papers (though local summarization is now default). To use these features, you need to configure your API keys.

1.  Create a copy of the `settings.env.example` file in the `configs` directory and name it `settings.env`:
    ```bash
    cp configs/settings.env.example configs/settings.env
    ```
2.  Add your API keys to the `configs/settings.env` file. You can obtain a Semantic Scholar API key from [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api).
    ```
    GEMINI_API_KEY="your-gemini-api-key"
    SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-api-key"
    ```

## Usage

To fetch and analyze papers, use the `arxiv` command:

```bash
literature-review arxiv "your query" [OPTIONS]
```

### Options

-   `--max-results`: The maximum number of papers to fetch (default: 10).
-   `--output-dir`: The directory to save the analysis results (default: `output`).
-   `--visualize-graph/--no-visualize-graph`: Generate a graph visualization of the papers (default: enabled).
-   `--sub-search/--no-sub-search`: Perform a sub-search on referenced articles (default: enabled). Requires `SEMANTIC_SCHOLAR_API_KEY` for full functionality.

### Example

```bash
literature-review arxiv "graph neural networks" --max-results 30 --output-dir gnn_analysis --no-sub-search
```

This command will:
1.  Fetch the top 30 papers related to "graph neural networks" from arXiv and Semantic Scholar.
2.  Analyze the papers and save the results to the `gnn_analysis` directory.
3.  Generate a similarity graph of the papers, highlighting the top 10 themes.
4.  *Not* perform a sub-search on referenced articles due to `--no-sub-search`.

## Outputs

The tool generates the following output files in the specified output directory:

-   `top_10_articles.csv`: A CSV file with the top 10 most relevant papers.
-   `sentiment_analysis.png`: A histogram of the sentiment scores of the paper abstracts.
-   `sentiment_summary.txt`: A text file with a summary of the sentiment analysis.
-   `wordcloud.png`: A word cloud generated from the paper titles and abstracts, with distinct colors for each word.
-   `cluster_summaries.json`: A JSON file with a summary of each paper cluster, including paper counts, top terms, and paper indices.
-   `graph_cluster_info.json`: A JSON file containing detailed information about each cluster in the similarity graph, including paper counts, top terms, and edge counts.
-   `similarity_graph.png`: A graph visualization of the paper similarity network, with nodes sized by cluster membership and a legend showing the top 10 themes with their edge counts.
-   `corpus_summary.txt`: A textual summary of the entire aggregated corpus of abstracts.
-   `reference_overlap.png`: (Only if `sub-search` is enabled and successful) A histogram showing the distribution of shared references between articles.

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