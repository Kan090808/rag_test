# Suggested Commands for RAG Agent Project

## Setup Commands
```bash
# Install dependencies
python setup_rag.py
# OR manually
pip install -r requirements.txt

# Copy configuration template
cp config_template.py config.py
```

## Running the Application
```bash
# Run main RAG agent (interactive mode)
python rag_agent.py

# Run demo script
python demo_rag.py

# Test RAG search functionality (without API calls)
python test_rag_search.py
```

## Development Commands
```bash
# Check project structure
ls -la

# View configuration
cat config.py

# Check log files
tail -f log.txt
tail -f test_log.txt

# Test configuration
python test_config.py
```

## File Management
```bash
# List data files
ls *.csv *.txt

# Check embeddings cache
ls -la embeddings.pkl

# Clean up logs
rm *.log *.txt
```

## Git Commands (Darwin/macOS)
```bash
git status
git add .
git commit -m "message"
git push
```

## System Utilities (Darwin/macOS)
```bash
# Find files
find . -name "*.py" -type f
find . -name "*.csv" -type f

# Search in files  
grep -r "pattern" .
grep -n "function_name" *.py

# Directory navigation
pwd
ls -la
cd path/to/directory
```