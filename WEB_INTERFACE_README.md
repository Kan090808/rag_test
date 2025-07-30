# RAG Agent - Unified Interface

A comprehensive RAG (Retrieval-Augmented Generation) Agent with both web and command-line interfaces for interacting with your knowledge base.

## Features

- 🌐 **Modern Web Interface**: Clean, responsive design that works on desktop and mobile
- � **Command-Line Interface**: Traditional terminal-based interaction mode
- �💬 **Real-time Chat**: Interactive conversation interface with conversation history
- 🤖 **Intelligent Responses**: AI-powered responses based on your knowledge base
- 📚 **Knowledge Base Integration**: Supports TXT files for document knowledge
- 🔄 **Conversation Management**: Clear history, show history, and persistent conversations
- 📊 **System Status**: Real-time status monitoring of the RAG agent
- 🎨 **Responsive Design**: Works seamlessly on all device sizes

## Quick Start

### Method 1: Web Interface (Recommended)

1. **Run the startup script**:
   ```bash
   python start_web.py
   ```
   
   OR run directly:
   ```bash
   python app.py
   ```

2. **Access the interface**:
   Open your browser and go to: `http://localhost:5001`

### Method 2: Command-Line Interface

1. **Run in CLI mode**:
   ```bash
   python app.py --cli
   ```

   This provides the traditional terminal-based chat interface.

### Method 3: Automatic Setup

1. **Use the startup script** (handles everything automatically):
   ```bash
   python start_web.py
   ```

   This script will:
   - Install all required dependencies
   - Check your configuration
   - Verify data files exist
   - Start the web interface

## Configuration

Make sure your `config.py` is properly configured:

```python
# API Configuration
OPENROUTER_API_KEY = "your-api-key-here"
MODEL_NAME = "openai/gpt-4o-mini"

# Data Files
TXT_FILES = [
    "your-document.txt"
]

# RAG Settings
TOP_K_DOCUMENTS = 10
MAX_TOKENS = 1000
TEMPERATURE = 0.7
```

## Usage

### Basic Chat
- Type your questions in the input field
- Press Enter or click "發送" to send
- The AI will respond based on your knowledge base

### Special Commands
- **clear_history**: Clear conversation history
- **show_history**: Display conversation history

### Control Buttons
- **清除歷史**: Clear conversation history
- **顯示歷史**: Show conversation history  
- **刷新狀態**: Refresh system status

## File Structure

```
rag/
├── app.py                 # Main application (web + CLI modes)
├── start_web.py          # Startup script (recommended)
├── rag_agent.py          # Core RAG agent class
├── config.py             # Configuration file
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface template
└── your-data-files.txt   # Your knowledge base files
```

## Usage Modes

### Web Interface Mode (Default)
```bash
python app.py
# Opens web interface at http://localhost:5001
```

### Command-Line Mode
```bash
python app.py --cli
# Starts terminal-based chat interface
```

### Using Startup Script (Easiest)
```bash
python start_web.py
# Handles setup and starts web interface
```

## Requirements

- Python 3.7+
- Flask 2.0+
- All dependencies listed in `requirements.txt`

## Troubleshooting

### Common Issues

1. **"RAG Agent not initialized" error**:
   - Check that your data files exist
   - Verify your config.py settings
   - Ensure your API key is valid

2. **"No documents loaded" warning**:
   - Make sure your TXT files exist in the correct location
   - Check the file paths in your config.py

3. **Import errors**:
   - Run `pip install -r requirements.txt` to install dependencies
   - Make sure you're using Python 3.7+

4. **Port already in use**:
   - Change the port in `app.py`: `app.run(port=5002)`
   - Or kill the process using port 5001

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all files are in the correct locations
3. Ensure your configuration is correct
4. Check that all dependencies are installed

## Advanced Usage

### Customizing the Interface

You can modify `templates/index.html` to customize:
- Colors and styling
- Layout and structure
- Additional features
- Language and text

### Adding New Features

The Flask app (`app.py`) can be extended with:
- File upload functionality
- Multiple knowledge bases
- User authentication
- Chat export features

### API Endpoints

The web interface provides these API endpoints:

- `GET /`: Main chat interface
- `POST /chat`: Send chat messages
- `GET /status`: System status information
- `POST /clear_history`: Clear conversation history

## License

This project is part of the RAG Agent system. Please refer to the main project license.
