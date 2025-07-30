# Configuration file for RAG Agent
# Copy this file to config.py and fill in your actual API key

OPENROUTER_API_KEY = "your-openrouter-api-key-here"

# Model configuration
MODEL_NAME = "anthropic/claude-3.5-haiku"

# RAG configuration
TOP_K_DOCUMENTS = 3
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# File paths
CSV_FILES = [
    "AmazonBedrock_20250217FAQ.csv",
    "AmazonBedrock_20250217additional.csv",
    "AmazonBedrock_2025021404.csv"
]

EMBEDDINGS_FILE = "embeddings.pkl"

# Logging configuration
LOG_FILE = "log.txt"
ENABLE_LOGGING = True
