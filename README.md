## Team

- **M. Dhana Kamakshi** — Project development, implementation, integration, evaluation, and documentation
- **Sai Abhijna P** — Project development and implementation
- **Mamatha Pandiri** — Project development and implementation


# Adaptive Trust Layer (ATL)

Middleware that sits between users and multiple LLMs. It analyzes each query, routes it to a suitable model, retrieves supporting documents, verifies the answer's claims with an NLI model, and returns a confidence score with a full agent trace.

## Architecture

The Adaptive Trust Layer consists of:

- Streamlit UI
- FastAPI backend
- LangGraph orchestration
- Analyzer agent
- Planner agent
- Router agent
- Retriever
- Generator
- NLI-based Verifier
- Confidence scoring
- Ollama and Groq LLMs
- ChromaDB vector database

## Setup

1. Install Python 3.11.
2. Create and activate a virtual environment:

   `python -m venv venv`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file and add the required Groq API key.
5. Install Ollama and pull the local model:

   `ollama pull llama3.2:3b`

6. Add documents to `data/docs/` and run ingestion if required.

## Run

### Backend

`uvicorn app.main:app --reload`

### Streamlit UI

`streamlit run ui/streamlit_app.py`

## Evaluation

Generate the evaluation dataset:

`python -m eval.make_dataset`

Run the evaluation:

`python -m eval.run_eval --systems A_small B_large C_rag_medium D_atl`

Analyze the results:

`python -m eval.analyze`

## Results

Evaluation results and generated analysis files are stored in:

`eval/results/`

## Team

Add team members and their respective roles here.