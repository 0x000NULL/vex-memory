# Contributing to Vex Memory

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/0x000NULL/vex-memory.git
cd vex-memory
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the database
docker compose up db -d

# Run the API locally
uvicorn api:app --reload
```

## Running Tests

```bash
pytest tests/ -v
```

## Making Changes

1. Fork the repo and create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all existing tests pass
4. Keep commits focused and messages clear
5. Open a PR with a description of what and why

## Code Style

- Python: follow PEP 8, use type hints where practical
- SQL: uppercase keywords, lowercase identifiers
- Keep functions focused and well-documented

## Areas for Contribution

- **Embedding model support** — add adapters for OpenAI, Cohere, or local models beyond Ollama
- **Retrieval strategies** — new search algorithms or hybrid approaches
- **Dashboard features** — visualizations, graph explorer, search UI
- **Performance** — query optimization, caching, batch operations
- **Documentation** — examples, tutorials, integration guides

## Reporting Issues

Open a GitHub issue with:
- What you expected vs. what happened
- Steps to reproduce
- Environment details (OS, Python version, Docker version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
