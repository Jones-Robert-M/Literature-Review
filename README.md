# Literature Review

An autonomous literature review tool that uses large language models to help you conduct literature reviews.

## Installation

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

## Configuration

This tool can use the Gemini API to summarize papers. To use this feature, you need to configure your Gemini API key.

1.  Create a copy of the `settings.env.example` file in the `configs` directory and name it `settings.env`:
    ```bash
    cp configs/settings.env.example configs/settings.env
    ```
2.  Add your Gemini API key to the `configs/settings.env` file:
    ```
    GEMINI_API_KEY="your-api-key"
    ```

## Usage

### arXiv

To fetch papers from arXiv, use the `arxiv` command:

```bash
literature-review arxiv "your query" --max-results <number-of-results>
```

-   `"your query"`: The search query for arXiv.
-   `--max-results`: The maximum number of papers to fetch (optional, defaults to 10).

If you have configured your Gemini API key, the tool will also provide a summary of each paper's abstract.

## Project Structure

```
.
├── configs
│   └── settings.env.example
├── src
│   └── literature_review
│       ├── __init__.py
│       ├── arxiv_fetcher.py
│       ├── core.py
│       ├── main.py
│       └── semantic_scholar_fetcher.py
├── pyproject.toml
└── README.md
```

-   `configs`: Contains configuration files.
-   `src/literature_review`: The main source code for the project.
-   `pyproject.toml`: Defines the project structure and dependencies.
-   `README.md`: This file.
