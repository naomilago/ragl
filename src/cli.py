'''
  Command-line interface for Milestone 1.

  Usage:
    uv run python -m src.cli ingest data/sample.pdf
    uv run python -m src.cli chat
'''

import typer

from src.rag_pipeline import RagPipeline

app = typer.Typer()


@app.command()
def ingest(pdf_path: str):
  ''' Load a PDF into the vector database (wipes any previous document). '''
  pipeline = RagPipeline()
  n_chunks = pipeline.pdf_ingester(pdf_path)
  typer.echo(f'Ingested {n_chunks} chunks from {pdf_path}')


@app.command()
def chat():
  ''' Start an interactive chat loop against the last ingested PDF. '''
  pipeline = RagPipeline()
  typer.echo("Chat started. Type 'exit' to quit.\n")
  while True:
    question = typer.prompt('You')
    if question.strip().lower() in ('exit', 'quit'):
      break
    result = pipeline.question_asker(question)
    typer.echo(f"\nAssistant: {result['answer']}")
    typer.echo(f"(sources: pages {result['pages']})\n")


if __name__ == '__main__':
  app()