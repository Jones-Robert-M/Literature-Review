import click
from dotenv import load_dotenv
import os
import google.generativeai as genai
from literature_review.arxiv_fetcher import fetch_arxiv

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

@click.group()
def cli():
    """An autonomous literature review tool."""
    if not os.getenv("GEMINI_API_KEY"):
        click.echo("Warning: GEMINI_API_KEY not found. Summarization will be disabled.")

@cli.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum number of papers to fetch.')
def arxiv(query, max_results):
    """Fetches papers from arXiv and summarizes them."""
    click.echo(f"Fetching papers from arXiv with query: '{query}' and max results: {max_results}")
    papers = fetch_arxiv(query, max_results=max_results)
    
    if not papers:
        click.echo("No papers found.")
        return
        
    model = get_gemini_model()
    
    for paper in papers:
        click.echo(f"\nTitle: {paper['title']}")
        click.echo(f"Authors: {paper['authors']}")
        click.echo(f"URL: {paper['url']}")
        
        if model:
            prompt = f"Summarize the following abstract in 3 sentences:\n\n{paper['abstract']}"
            
            try:
                response = model.generate_content(prompt)
                click.echo(f"Summary: {response.text}")
            except Exception as e:
                click.echo(f"Error summarizing paper: {e}")

if __name__ == '__main__':
    cli()
