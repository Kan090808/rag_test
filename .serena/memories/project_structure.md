# Project Structure

## Main Files
- **rag_agent.py**: Main RAG agent implementation with RAGAgent class
- **config.py**: Configuration file (copied from config_template.py)
- **setup_rag.py**: Setup script for dependencies and file checking
- **demo_rag.py**: Demo script for testing the agent

## Configuration Files
- **config_template.py**: Template for configuration
- **requirements.txt**: Python dependencies

## Data Files
- **CSV files**: FAQ data in Chinese with '問題' and '答案' columns
  - AmazonBedrock_20250217FAQ.csv
  - AmazonBedrock_20250217additional.csv  
  - AmazonBedrock_2025021404.csv
- **TXT files**: Q&A data in "Q: [question] A: [answer]" format
  - 0708.txt

## Generated Files
- **embeddings.pkl**: Cached sentence embeddings
- **log.txt**: Application logs
- **test_log.txt**: Test operation logs

## Test Files
- **test_rag_search.py**: Test semantic search without API calls
- **test_config.py**: Configuration validation
- **test_agent_logging.py**: Logging functionality tests

## Directory Structure
```
/rag/
├── .serena/           # Serena configuration
├── rag_agent.py       # Main implementation
├── config.py          # Configuration
├── setup_rag.py       # Setup script
├── requirements.txt   # Dependencies
├── *.csv             # CSV data files
├── *.txt             # TXT data files
├── embeddings.pkl    # Cached embeddings
└── *.log             # Log files
```

## Key Modules
- **RAGAgent class**: Main functionality in rag_agent.py
- **Configuration**: Centralized in config.py with arrays for CSV_FILES and TXT_FILES
- **Dual file format support**: Handles both CSV and TXT files seamlessly