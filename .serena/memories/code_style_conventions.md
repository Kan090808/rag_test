# Code Style and Conventions

## Python Style Guidelines
- **PEP 8 compliance**: Follow standard Python style guide
- **Line length**: Keep lines under 100 characters where reasonable
- **Indentation**: 4 spaces (no tabs)

## Naming Conventions
- **Functions/Variables**: snake_case (`load_csv_data`, `embedding_model`)
- **Classes**: PascalCase (`RAGAgent`)
- **Constants**: UPPER_SNAKE_CASE (`TOP_K_DOCUMENTS`, `OPENROUTER_API_KEY`)
- **Private methods**: Leading underscore (`_setup_logging`, `_process_qa_pair`)

## Type Hints
- Use type hints for function parameters and return values:
```python
def load_csv_data(self, csv_file_path: str, question_col: str = '問題', answer_col: str = '答案') -> int:
    """Load data from CSV file"""
```
- Import typing utilities: `from typing import List, Dict, Any, Optional`

## Docstrings
- Use triple quotes for docstrings
- Include parameter descriptions and return values:
```python
def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using semantic similarity
    
    Args:
        query: The search query
        top_k: Number of documents to return (uses default if None)
        
    Returns:
        List of relevant documents with metadata
    """
```

## Error Handling
- Use try-except blocks for external operations (file I/O, API calls)
- Log errors appropriately
- Provide meaningful error messages to users

## Configuration Management
- Store configuration in `config.py`
- Use `getattr()` with defaults for optional config values
- Keep sensitive data (API keys) in config files, not in code