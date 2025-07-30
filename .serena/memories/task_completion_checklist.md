# Task Completion Guidelines

## When a Task is Completed

### 1. Code Validation
- **Test the functionality**: Run the main application to ensure it works
- **Check imports**: Verify all imports are available and correct
- **Validate configuration**: Ensure config.py has the correct structure

### 2. Testing Commands
```bash
# Test the RAG search without API calls
python test_rag_search.py

# Test the full RAG agent (requires API key)
python rag_agent.py

# Run setup script to verify dependencies
python setup_rag.py
```

### 3. File Validation
- **Check data files exist**: Verify CSV and TXT files are in place
- **Validate file formats**: 
  - CSV files should have '問題' and '答案' columns
  - TXT files should use "Q: [question] A: [answer]" format
- **Check generated files**: Verify embeddings.pkl is created correctly

### 4. Configuration Review
- **API key setup**: Ensure OPENROUTER_API_KEY is configured
- **File arrays**: Verify CSV_FILES and TXT_FILES contain correct file names
- **System prompt**: Check SYSTEM_PROMPT is properly configured

### 5. Logging Verification
- **Check log outputs**: Review log.txt for any errors or warnings
- **Test logging**: Ensure logging operations work correctly
- **Clean up**: Remove test log files if needed

### 6. Documentation Updates
- **Update README.md**: If functionality changes significantly
- **Code comments**: Ensure complex functions have appropriate docstrings
- **Configuration docs**: Update config examples if new options added

### 7. No Specific Linting/Formatting Tools
- The project doesn't use specific linting tools like black, flake8, or pylint
- Manual code review for PEP 8 compliance is sufficient
- Focus on readability and maintainability