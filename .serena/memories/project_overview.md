# RAG Agent Project

## Purpose
A Python-based Retrieval-Augmented Generation (RAG) agent that uses semantic search to find relevant information from CSV and TXT files and generates responses using OpenRouter's Claude models.

## Key Features
- **Semantic Search**: Uses sentence transformers to find relevant information
- **AI-Powered Responses**: Generates answers using Claude models via OpenRouter
- **Multi-format Support**: Loads FAQ data from both CSV files and TXT files (Q: A: format)
- **Embedding Cache**: Saves embeddings to avoid recomputation
- **Interactive Chat**: Command-line interface for conversations
- **Configurable System Prompts**: Customizable AI behavior through configuration

## Tech Stack
- **Python 3.x**: Main programming language
- **sentence-transformers**: For creating semantic embeddings
- **pandas**: CSV file handling  
- **numpy**: Numerical computations
- **scikit-learn**: Similarity calculations
- **requests**: API calls to OpenRouter
- **torch**: Required by sentence-transformers
- **transformers**: NLP models