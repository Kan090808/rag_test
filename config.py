# Configuration file for RAG Agent
# Copy this file to config.py and fill in your actual API key

OPENROUTER_API_KEY = "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"

# Model configuration
# MODEL_NAME = "anthropic/claude-3.5-haiku"
# MODEL_NAME = "openai/gpt-3.5-turbo"
MODEL_NAME = "openai/gpt-4.1-mini"
# MODEL_NAME = "google/gemini-2.5-flash" #效果不好


# RAG configuration
TOP_K_DOCUMENTS = 10
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# File paths
CSV_FILES = [
    # "AmazonBedrock_20250217FAQ.csv",
    # "AmazonBedrock_20250217additional.csv",
    # "AmazonBedrock_2025021404.csv"
]

TXT_FILES = [
    "0708.txt"
]

EMBEDDINGS_FILE = "embeddings.pkl"

# System prompt for the RAG agent
SYSTEM_PROMPT = """
1.如果使用者用什麼國家語系詢問問題，請使用相同語系回覆。
2.資料庫包含所有問題及正確答案，請根據資料庫提供準確的資訊給用戶。
3.若資料庫中無法找到相關答案，必須一字不差的回覆：「很抱歉，阿萊Eli 可能無法完全理解您的問題。您可以嘗試用不同方式重新描述，或在聊天室中輸入「轉接真人客服」，也可以直接點擊下方的轉接真人客服按鈕獲得協助。」
4.請注意，不要回答與Letstalk不相關的答案。
"""

# Logging configuration
LOG_FILE = "log.txt"
ENABLE_LOGGING = True
