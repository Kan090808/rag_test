# 🤖 RAG Agent

[![Deploy to GitHub Pages](https://github.com/Kan090808/rag_test/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/Kan090808/rag_test/actions/workflows/deploy-pages.yml)

A sophisticated Retrieval-Augmented Generation (RAG) system with conversation memory and enhanced search capabilities.

## 🌐 Live Documentation

Visit the automatically deployed documentation site: **[https://kan090808.github.io/rag_test](https://kan090808.github.io/rag_test)**

The documentation site is automatically updated with every commit to the master branch.

## ✨ Features

- 🔍 **Multi-stage retrieval** - Combines keyword and semantic search
- 💬 **Conversation memory** - Maintains context across interactions  
- 🎯 **Query expansion** - Intelligent query enhancement with negation handling
- 📚 **FAQ support** - Load and search Q&A pairs from text files
- 🌐 **Web interface** - Clean, responsive UI for easy interaction
- 🚀 **Auto-deployment** - GitHub Pages deployment on every commit

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kan090808/rag_test.git
   cd rag_test
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   cp config_template.py config.py
   # Edit config.py and add your OpenRouter API key
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 📁 Project Structure

```
rag_test/
├── rag_agent.py          # Main RAG agent implementation
├── app.py                # Flask web application
├── config_template.py    # Configuration template
├── requirements.txt      # Python dependencies
├── templates/            # Web interface templates
├── docs/                 # GitHub Pages documentation
└── .github/workflows/    # GitHub Actions for auto-deployment
```

## 🛠️ Configuration

Copy `config_template.py` to `config.py` and configure:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `MODEL_NAME`: LLM model to use (default: claude-3-haiku)
- `MAX_TOKENS`: Maximum response tokens
- `TEMPERATURE`: Response creativity (0.0-1.0)

## 📖 Documentation

- [Web Interface Guide](WEB_INTERFACE_README.md)
- [Agent Summary](AGENT_SUMMARY.md) 
- [Search Improvement Report](RAG_SEARCH_IMPROVEMENT_REPORT.md)

## 🔄 Auto-Deployment

This project uses GitHub Actions to automatically deploy documentation to GitHub Pages on every commit. The workflow:

1. Builds the project
2. Generates static documentation
3. Deploys to GitHub Pages
4. Updates the live site at `https://kan090808.github.io/rag_test`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit and push
5. The documentation site will automatically update!

## 📄 License

This project is open source and available under the MIT License.

---

**Live Site**: https://kan090808.github.io/rag_test
