# NotebookLM-Style Hybrid RAG System

A production-ready, fully local Retrieval-Augmented Generation (RAG) system powered by FastAPI, PostgreSQL, Redis, Qdrant, and LLaMA.cpp.

## 📖 Documentation
Please refer to the comprehensive [walkthrough.md](./walkthrough.md) for the complete architecture breakdown, getting started guide, and usage instructions.

## 🚀 Quick Start
```bash
# Create .env and add optional HF_TOKEN
echo "HF_TOKEN=your_token_here" > .env

# Boot the system
docker-compose up -d --build
```
Access the React Dashboard at http://localhost:3000.
