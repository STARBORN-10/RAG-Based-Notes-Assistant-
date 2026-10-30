# RAG-Based-Notes-Assistant

An AI-powered Notes Assistant built using Retrieval-Augmented Generation (RAG).

## Features
- Load and process study notes.
- Support PDF and text documents.
- Generate embeddings from document content.
- Retrieve relevant information using vector search.
- Generate context-aware answers using an LLM.

## Tech Stack
- Python
- LangChain
- Hugging Face
- FAISS
- OpenAI
- PyPDF

## Project Structure

```text
RAG-Based-Notes-Assistant/
│
├── Document Loaders/
│   ├── pdf.py
│   └── test.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore