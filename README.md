# Literature Review

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

An autonomous literature review tool that uses large language models to help you conduct literature reviews.

## Features

-   **Fetch Papers**: Fetches papers from arXiv and Semantic Scholar.
-   **First-Order Sub-Search**: Automatically performs a sub-search on articles referenced by the initial query (requires Semantic Scholar API Key).
-   **In-depth Analysis**: Performs sentiment analysis, clustering, and ranking of papers.
-   **Corpus Summary**: Generates a textual summary of the entire aggregated corpus of abstracts.
-   **Visualizations**: Generates word clouds and network graphs to visualize the relationships between papers. Node sizes in the network graph correspond to the number of papers in each cluster, and the plot is designed for better visual appeal.
-   **Local Summarization**: Uses a local summarization model to summarize the abstracts of the top papers.
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
-   `--sub-search/--no-sub-search`: Perform a sub-search on referenced articles (default: enabled). Requires `SEMANTIC_SCHOLAR_API_KEY`.

### Example

```bash
literature-review arxiv "graph neural networks" --max-results 30 --output-dir gnn_analysis --no-sub-search
```

This command will:
1.  Fetch the top 30 papers related to "graph neural networks" from arXiv and Semantic Scholar.
2.  Analyze the papers and save the results to the `gnn_analysis` directory.
3.  Generate a similarity graph of the papers.
4.  *Not* perform a sub-search on referenced articles due to `--no-sub-search`.

## Outputs

The tool generates the following output files in the specified output directory:

-   `top_10_articles.csv`: A CSV file with the top 10 most relevant papers.
-   `sentiment_analysis.png`: A histogram of the sentiment scores of the paper abstracts.
-   `sentiment_summary.txt`: A text file with a summary of the sentiment analysis.
-   `wordcloud.png`: A word cloud generated from the paper titles and abstracts.
-   `cluster_summaries.json`: A JSON file with a summary of each paper cluster.
-   `similarity_graph.png`: A graph visualization of the paper similarity network.
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
