import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
from typing import List, Dict, Any, Optional, Tuple
import pickle
import logging
from datetime import datetime
import config

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # Optional dependency

try:
    import jieba as _jieba
except Exception:
    _jieba = None


def _tokenize_zh(text: str) -> List[str]:
    """Tokenize Chinese text using jieba when available, else fallback to character-level tokens."""
    if _jieba is not None:
        try:
            return list(_jieba.cut(text))
        except Exception:
            pass
    # Fallback: naive split into characters (removes spaces)
    return [ch for ch in text if not ch.isspace()]


class RAGAgent:
    def __init__(self, openrouter_api_key: str, model_name: Optional[str] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None, log_file: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        """
        Initialize the RAG Agent

        Args:
            openrouter_api_key: OpenRouter API key
            model_name: Model to use (uses config value if None)
            max_tokens: Maximum tokens for response generation (uses config value if None)
            temperature: Temperature for response generation (uses config value if None)
            log_file: Path to log file for tracking operations (uses config value if None)
            system_prompt: System prompt to guide the AI's responses (uses config value if None)
        """
        self.openrouter_api_key = openrouter_api_key
        self.model_name = model_name if model_name is not None else config.MODEL_NAME
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.system_prompt = system_prompt if system_prompt is not None else config.SYSTEM_PROMPT
        self.default_top_k = config.TOP_K_DOCUMENTS
        self.log_file = log_file if log_file is not None else config.LOG_FILE
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        # Setup logging
        self._setup_logging()

        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.log_operation("SYSTEM", "Loading sentence transformer model", {
            "multilingual": getattr(config, 'USE_MULTILINGUAL_EMBEDDINGS', False),
            "embedding_model": getattr(config, 'EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        })
        if getattr(config, 'USE_MULTILINGUAL_EMBEDDINGS', False):
            self._embedding_is_e5 = True
            self.embedding_model = SentenceTransformer(getattr(config, 'EMBEDDING_MODEL_NAME', 'intfloat/multilingual-e5-base'))
        else:
            self._embedding_is_e5 = False
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Storage for documents and embeddings
        self.documents: List[str] = []
        self.embeddings = None
        self.metadata: List[Dict[str, Any]] = []
        # Chunk-level storage and BM25
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.chunk_embeddings = None
        self._bm25 = None

        # Conversation history management
        self.conversation_history = []  # List of {"user": str, "assistant": str, "timestamp": str}
        self.max_history_length = 20
        self.conversation_summary = ""
        self.conversation_turns_in_prompt = int(getattr(config, 'CONVERSATION_TURNS_IN_PROMPT', 4))

        # Query expansion configuration for Letstalk app
        self.query_expansions = {
            # 時刻相關擴展
            "時刻": ["時刻功能", "時刻貼文", "發布時刻", "分享時刻", "時刻頁面", "時刻內容", "我的時刻"],
            "找不到時刻": ["時刻功能", "時刻", "沒有時刻", "時刻不見", "時刻消失", "為什麼沒有時刻", "時刻在哪裡", "時刻不可見"],
            "看不到時刻": ["時刻功能", "時刻", "時刻不可見", "時刻隱藏", "為什麼看不到時刻", "時刻介面"],
            "時刻不見": ["時刻功能", "時刻", "沒有時刻", "時刻消失", "為什麼沒有時刻功能", "時刻不可見"],

            # 其他功能擴展
            "群": ["群組", "群聊", "群組聊天", "建立群組", "群組功能", "群組成員"],
            "好友": ["好友", "朋友", "聯絡人", "加好友", "好友列表", "好友管理"],
            "訊息": ["訊息", "消息", "聊天", "對話", "傳送", "接收"],
            "登入": ["登入", "登錄", "登入問題", "無法登入", "登入失敗"],
            "帳號": ["帳號", "帳戶", "用戶", "註冊", "帳號管理", "帳號設定"],
            "檔案": ["檔案", "文件", "附件", "圖片", "照片", "影片"],
            "設定": ["設定", "設置", "配置", "選項", "偏好設定"],
            "通知": ["通知", "提醒", "推播", "通知設定", "消息提醒"],
            "封鎖": ["封鎖", "阻擋", "黑名單", "封鎖好友", "解除封鎖"],
            "下載": ["下載", "載入", "保存", "儲存", "無法下載"],
            "刪除": ["刪除", "移除", "清除", "刪掉", "消除"],
            "密碼": ["密碼", "密碼重設", "忘記密碼", "修改密碼", "密碼問題"],
            "聊天": ["聊天", "對話", "談話", "交談", "聊天室"],
            "功能": ["功能", "特色", "選項", "工具", "服務"],
            "問題": ["問題", "錯誤", "故障", "異常", "無法使用"],
            "安裝": ["安裝 letstalk", "letstalk 安裝檔", "letstalk 安裝包", "更新 letstalk", "letstalk 更新檔"],

            # 問題導向擴展
            "找不到": ["無法找到", "看不到", "不見了", "消失", "沒有", "不存在"],
            "無法": ["不能", "不會", "無法使用", "失效", "故障"],
            "沒有": ["找不到", "不見", "消失", "不存在", "看不到"],
            "圖片": ["圖片"],
        }

        print("RAG Agent initialized successfully!")
        self.log_operation("SYSTEM", "RAG Agent initialized", {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        })

    # --- Logging helpers ---
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("=== RAG Agent Log Started ===\n")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")

    def log_operation(self, operation_type: str, operation: str, details: Dict[str, Any]):
        """
        Log operation details to file

        Args:
            operation_type: Type of operation (SEARCH, ANSWER, SYSTEM, etc.)
            operation: Description of the operation
            details: Additional details to log
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "type": operation_type,
                "operation": operation,
                "details": details
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{json.dumps(log_entry, ensure_ascii=False, indent=2)}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    def change_model(self, new_model_name: str) -> bool:
        """Change the model used for generating responses."""
        try:
            old_model = self.model_name
            self.model_name = new_model_name
            self.log_operation("SYSTEM", "Model changed", {
                "old_model": old_model,
                "new_model": new_model_name
            })
            print(f"Model changed from '{old_model}' to '{new_model_name}'")
            return True
        except Exception as e:
            print(f"Error changing model: {e}")
            self.log_operation("ERROR", "Failed to change model", {
                "old_model": getattr(self, 'model_name', None),
                "new_model": new_model_name,
                "error": str(e)
            })
            return False

    # --- Data loading ---
    def load_txt_data(self, txt_file_path: str):
        """
        Load FAQ data from txt file in Q: A: format

        Args:
            txt_file_path: Path to txt file
        """
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by Q: to get each Q&A pair, but keep Q: at the beginning
            lines = content.split('\n')
            current_qa = ""
            qa_count = 0

            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    if current_qa:
                        self._process_qa_pair(current_qa, txt_file_path, qa_count)
                        qa_count += 1
                    current_qa = line
                elif line and current_qa:
                    current_qa += " " + line

            if current_qa:
                self._process_qa_pair(current_qa, txt_file_path, qa_count)
                qa_count += 1

            print(f"Loaded {qa_count} Q&A pairs from {txt_file_path}")
            self.log_operation("DATA_LOAD", "Loaded TXT file", {
                "file": txt_file_path,
                "qa_pairs": qa_count
            })
        except Exception as e:
            print(f"Error loading TXT file {txt_file_path}: {e}")
            self.log_operation("ERROR", "Failed to load TXT file", {
                "file": txt_file_path,
                "error": str(e)
            })

    def _process_qa_pair(self, qa_text: str, source_file: str, index: int):
        """Process a Q&A pair from TXT format"""
        if ' A:' in qa_text:
            question_part, answer_part = qa_text.split(' A:', 1)
            question = question_part.replace('Q:', '').strip()
            answer = answer_part.strip()
            if question and answer:
                content = f"問題：{question}\n答案：{answer}"
                self.documents.append(content)
                self.metadata.append({
                    'source': source_file,
                    'index': index,
                    'question': question,
                    'answer': answer
                })

    def load_multiple_txt_files(self, txt_files: List[str]):
        """Load multiple TXT files"""
        for txt_file in txt_files:
            if os.path.exists(txt_file):
                self.load_txt_data(txt_file)
            else:
                print(f"File not found: {txt_file}")

    def load_multiple_files(self, txt_files: Optional[List[str]] = None):
        """Load multiple TXT files"""
        if txt_files:
            self.load_multiple_txt_files(txt_files)

    # --- Chunking and encoding ---
    def _maybe_chunk_documents(self):
        """Split documents into overlapping character chunks for finer retrieval."""
        if self.chunks and self.chunk_metadata:
            return
        enable_chunk = getattr(config, 'ENABLE_CHUNKING', True)
        size = int(getattr(config, 'CHUNK_CHAR_SIZE', 350))
        overlap = int(getattr(config, 'CHUNK_CHAR_OVERLAP', 40))
        if not enable_chunk:
            self.chunks = list(self.documents)
            self.chunk_metadata = list(self.metadata)
            return
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            text = doc
            start = 0
            c_id = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                m = dict(meta)
                m.update({
                    'chunk_id': c_id,
                    'char_start': start,
                    'char_end': end
                })
                metas.append(m)
                if end == len(text):
                    break
                start = end - overlap
                c_id += 1
        self.chunks = chunks
        self.chunk_metadata = metas

    def _build_bm25(self):
        """Build BM25 index for chunks using jieba tokenization."""
        if not getattr(config, 'ENABLE_BM25', True):
            return
        if BM25Okapi is None:
            self._bm25 = None
            return
        if not self.chunks:
            self._maybe_chunk_documents()
        tokenized = [_tokenize_zh(text) for text in self.chunks]
        try:
            self._bm25 = BM25Okapi(tokenized)
        except Exception as e:
            self._bm25 = None
            self.log_operation("WARN", "BM25 build exception", {"error": str(e)})

    def _encode_texts(self, texts: List[str]):
        """Encode texts; if using E5-family, add prompts and normalize."""
        if getattr(self, '_embedding_is_e5', False):
            passages = [f"passage: {t}" for t in texts]
            return self.embedding_model.encode(passages, normalize_embeddings=True)
        return self.embedding_model.encode(texts, normalize_embeddings=True)

    def _encode_query(self, query: str):
        if getattr(self, '_embedding_is_e5', False):
            return self.embedding_model.encode([f"query: {query}"], normalize_embeddings=True)
        return self.embedding_model.encode([query], normalize_embeddings=True)

    # --- Embeddings persistence ---
    def create_embeddings(self, save_path: str = "embeddings.pkl"):
        """
        Create embeddings for all documents

        Args:
            save_path: Path to save embeddings
        """
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return

        print("Creating embeddings for documents...")
        self._maybe_chunk_documents()
        self.chunk_embeddings = self._encode_texts(self.chunks)
        try:
            self.embeddings = self._encode_texts(self.documents)
        except Exception:
            self.embeddings = None
        try:
            self._build_bm25()
        except Exception as e:
            self.log_operation("WARN", "BM25 build failed", {"error": str(e)})

        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'documents': self.documents,
                'metadata': self.metadata,
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'chunk_embeddings': self.chunk_embeddings
            }, f)

        print(f"Embeddings created and saved to {save_path}")

    def load_embeddings(self, load_path: str = "embeddings.pkl"):
        """Load pre-computed embeddings"""
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings')
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.chunks = data.get('chunks', []) or list(self.documents)
                self.chunk_metadata = data.get('chunk_metadata', []) or list(self.metadata)
                self.chunk_embeddings = data.get('chunk_embeddings')
                try:
                    self._build_bm25()
                except Exception as e:
                    self.log_operation("WARN", "BM25 rebuild failed on load", {"error": str(e)})
            print(f"Embeddings loaded from {load_path}")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    # --- Conversation memory ---
    def add_to_conversation_history(self, user_query: str, assistant_response: str):
        """Add a conversation pair to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_entry = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": timestamp
        }
        self.conversation_history.append(conversation_entry)
        if len(self.conversation_history) > self.max_history_length:
            self._manage_conversation_history()
        self.log_operation("CONVERSATION", "Added to conversation history", {
            "history_length": len(self.conversation_history),
            "has_summary": bool(self.conversation_summary)
        })

    def _manage_conversation_history(self):
        if len(self.conversation_history) <= self.max_history_length:
            return
        conversations_to_summarize = self.conversation_history[:self.max_history_length//2]
        old_summary = self._create_conversation_summary(conversations_to_summarize)
        if self.conversation_summary:
            combined_summary = f"{self.conversation_summary}\n\n--- 新的對話摘要 ---\n{old_summary}"
        else:
            combined_summary = old_summary
        self.conversation_summary = combined_summary
        self.conversation_history = self.conversation_history[self.max_history_length//2:]
        self.log_operation("CONVERSATION", "Conversation history summarized and trimmed", {
            "conversations_summarized": len(conversations_to_summarize),
            "remaining_history": len(self.conversation_history),
            "summary_length": len(self.conversation_summary)
        })

    def _create_conversation_summary(self, conversations: List[Dict[str, str]]) -> str:
        conversation_text = ""
        for i, conv in enumerate(conversations, 1):
            conversation_text += f"對話 {i}:\n"
            conversation_text += f"用戶: {conv['user']}\n"
            conversation_text += f"助手: {conv['assistant']}\n"
            conversation_text += f"時間: {conv['timestamp']}\n\n"
        summary_prompt = f"""請為以下對話歷史創建一個簡潔的摘要，重點關注：
1. 用戶主要詢問的問題類型
2. 已經解決的問題
3. 用戶的主要關注點
4. 任何重要的上下文信息

對話歷史：
{conversation_text}

請提供簡潔的摘要："""
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model_name,
            'messages': [{
                'role': 'user',
                'content': summary_prompt
            }],
            'max_tokens': 500,
            'temperature': 0.3
        }
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            summary = result['choices'][0]['message']['content']
            token_usage = {}
            if 'usage' in result:
                token_usage = {
                    'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                    'completion_tokens': result['usage'].get('completion_tokens', 0),
                    'total_tokens': result['usage'].get('total_tokens', 0)
                }
            self.log_operation("SUMMARY", "Created conversation summary", {
                "conversations_count": len(conversations),
                "summary_length": len(summary),
                "token_usage": token_usage
            })
            return summary
        except Exception as e:
            fallback_summary = f"摘要 {len(conversations)} 個對話，時間範圍：{conversations[0]['timestamp']} 至 {conversations[-1]['timestamp']}"
            self.log_operation("ERROR", "Failed to create AI summary, using fallback", {
                "error": str(e),
                "fallback_summary": fallback_summary
            })
            return fallback_summary

    def get_conversation_context(self) -> str:
        context_parts = []
        if self.conversation_summary:
            context_parts.append(f"對話摘要：\n{self.conversation_summary}")
        if self.conversation_history:
            recent_conversations = []
            for conv in self.conversation_history[-5:]:
                recent_conversations.append(f"用戶: {conv['user']}\n助手: {conv['assistant']}")
            if recent_conversations:
                context_parts.append(f"最近對話：\n" + "\n\n".join(recent_conversations))
        return "\n\n--- 分隔線 ---\n\n".join(context_parts) if context_parts else ""

    def clear_conversation_history(self):
        self.conversation_history = []
        self.conversation_summary = ""
        self.log_operation("CONVERSATION", "Conversation history cleared", {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # --- Query processing ---
    def expand_query(self, query: str) -> str:
        """
        改进的查询扩展策略 - 更精准，减少噪音
        """
        expanded_terms = set()
        query_lower = query.lower()
        max_expansion_terms = getattr(config, 'QUERY_EXPANSION_LIMIT', 3)
        is_negation_query = any(neg_word in query_lower for neg_word in ['找不到', '看不到', '沒有', '没有', '不見', '无法', '無法', '不能', '消失'])
        for key, terms in self.query_expansions.items():
            if key in query_lower:
                expanded_terms.update(terms[:max_expansion_terms])
                self.log_operation("QUERY_EXPANSION", "Precise expansion applied", {
                    "matched_key": key,
                    "expanded_terms": terms[:max_expansion_terms],
                    "query": query
                })
                break
        if is_negation_query and not expanded_terms:
            for neg_word in ['找不到', '看不到', '沒有', '没有', '不見']:
                if neg_word in query_lower:
                    after_neg = query_lower.split(neg_word)[-1].strip()
                    if after_neg:
                        for key, terms in self.query_expansions.items():
                            if after_neg in key or key in after_neg:
                                expanded_terms.update(terms[:max_expansion_terms])
                                expanded_terms.add(after_neg)
                                self.log_operation("QUERY_EXPANSION", "Negation-aware expansion", {
                                    "negation_word": neg_word,
                                    "extracted_object": after_neg,
                                    "expanded_terms": terms[:max_expansion_terms]
                                })
                                break
                    break
        if not expanded_terms and len(query.strip()) <= 3:
            for key, terms in self.query_expansions.items():
                if any(char in key for char in query_lower):
                    expanded_terms.update(terms[:2])
                    break
        if expanded_terms:
            expanded_terms.discard(query_lower)
            expanded_query = query + " " + " ".join(list(expanded_terms)[:max_expansion_terms])
            self.log_operation("QUERY_EXPANSION", "Limited expansion completed", {
                "original": query,
                "expanded": expanded_query,
                "terms_added": len(expanded_terms),
                "max_allowed": max_expansion_terms
            })
            return expanded_query
        return query

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '了', '也', '都', '這', '那', '可以', '如何', '什麼', '為什麼', '怎麼', '哪裡'}
        negation_words = {'找不到', '沒有', '不能', '無法', '不會', '看不到', '不見', '消失', '沒看到', '不存在'}
        query_lower = query.lower()
        keywords: List[str] = []
        for neg_word in negation_words:
            if neg_word in query_lower:
                parts = query_lower.split(neg_word)
                if len(parts) > 1:
                    after = parts[1].strip()
                    if after:
                        keywords.append(after)
                before = parts[0].strip()
                if before:
                    keywords.append(before)
        if not keywords:
            for term in query_lower.split():
                if term not in stop_words and len(term) > 1:
                    keywords.append(term)
        unique_keywords: List[str] = []
        for k in keywords:
            if k and k not in unique_keywords and k not in stop_words:
                unique_keywords.append(k)
        self.log_operation("KEYWORD_EXTRACTION", "Extracted keywords from query", {
            "original_query": query,
            "extracted_keywords": unique_keywords,
            "extraction_method": "negation_aware" if any(neg in query_lower for neg in negation_words) else "standard"
        })
        return unique_keywords if unique_keywords else [query.lower()]

    # --- Retrieval helpers ---
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        keyword_matches: List[Dict[str, Any]] = []
        keywords = self._extract_keywords(query)
        for idx, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            exact_matches = 0
            partial_matches = 0
            for keyword in keywords:
                if keyword in doc_lower:
                    exact_matches += 1
                elif len(keyword) > 2:
                    keyword_chars = set(keyword)
                    doc_chars = set(doc_lower)
                    if len(keyword_chars.intersection(doc_chars)) / max(len(keyword_chars), 1) > 0.7:
                        partial_matches += 0.5
            total_matches = exact_matches + partial_matches
            if total_matches > 0:
                keyword_score = total_matches / max(len(keywords), 1)
                if any(word in doc_lower for word in ['問題', '如何', '為什麼', '怎麼', '無法', '不能']):
                    keyword_score *= 1.2
                keyword_matches.append({
                    'index': idx,
                    'content': doc,
                    'metadata': self.metadata[idx],
                    'keyword_score': keyword_score,
                    'similarity': keyword_score,
                    'matched_keywords': [kw for kw in keywords if kw in doc_lower]
                })
        keyword_matches.sort(key=lambda x: x['keyword_score'], reverse=True)
        return keyword_matches[:top_k]

    def _broad_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        broad_matches: List[Dict[str, Any]] = []
        query_chars = set(query.lower().replace(' ', ''))
        for idx, doc in enumerate(self.documents):
            doc_chars = set(doc.lower().replace(' ', ''))
            if query_chars:
                char_overlap = len(query_chars.intersection(doc_chars)) / max(len(query_chars), 1)
                if char_overlap > 0.3:
                    broad_matches.append({
                        'index': idx,
                        'content': doc,
                        'metadata': self.metadata[idx],
                        'broad_score': char_overlap,
                        'similarity': char_overlap
                    })
        broad_matches.sort(key=lambda x: x['broad_score'], reverse=True)
        return broad_matches[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not getattr(config, 'ENABLE_BM25', True):
            return []
        if self._bm25 is None:
            try:
                self._build_bm25()
            except Exception as e:
                self.log_operation("WARN", "BM25 not available", {"error": str(e)})
                return []
        if self._bm25 is None:
            return []
        tokens = _tokenize_zh(query)
        scores = self._bm25.get_scores(tokens)
        if top_k >= len(scores):
            idxs = list(range(len(scores)))
        else:
            idxs = np.argpartition(scores, -top_k)[-top_k:]
        ranked = sorted([(int(i), float(scores[int(i)])) for i in idxs], key=lambda x: x[1], reverse=True)
        return ranked

    def _combine_search_results(self, bm25_results: List[Tuple[int, float]],
                                semantic_similarities: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        keyword_weight = getattr(config, 'KEYWORD_WEIGHT', 0.6)
        semantic_weight = getattr(config, 'SEMANTIC_WEIGHT', 0.4)
        semantic_threshold = getattr(config, 'SEMANTIC_THRESHOLD', 0.15)
        semantic_indices: List[Tuple[int, float]] = [
            (idx, float(score)) for idx, score in enumerate(semantic_similarities) if score >= semantic_threshold
        ]
        semantic_indices.sort(key=lambda x: x[1], reverse=True)
        doc_scores: Dict[int, float] = {}
        if bm25_results:
            bm25_max = max(s for _, s in bm25_results) or 1.0
            for idx, s in bm25_results:
                doc_scores[idx] = doc_scores.get(idx, 0.0) + (s / bm25_max) * keyword_weight
        for idx, s in semantic_indices[:max(top_k * 3, 50)]:
            doc_scores[idx] = doc_scores.get(idx, 0.0) + s * semantic_weight
        combined_results: List[Dict[str, Any]] = []
        for idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
            content = self.chunks[idx] if idx < len(self.chunks) else (self.documents[idx] if idx < len(self.documents) else "")
            combined_results.append({
                'content': content,
                'metadata': meta,
                'similarity': float(score),
                'has_keyword_match': any(i == idx for i, _ in bm25_results),
                'semantic_score': float(semantic_similarities[idx]) if idx < len(semantic_similarities) else 0.0
            })
        return combined_results

    # --- Retrieval orchestrator ---
    def retrieve_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.default_top_k
        if self.chunk_embeddings is None:
            if not self.documents:
                print("No documents loaded. Please load data first.")
                return []
            self._maybe_chunk_documents()
            self.chunk_embeddings = self._encode_texts(self.chunks)
            try:
                self._build_bm25()
            except Exception as e:
                self.log_operation("WARN", "BM25 build failed in retrieval", {"error": str(e)})
        self.log_operation("SEARCH", "Starting enhanced retrieval", {
            "original_query": query,
            "query_length": len(query.strip()),
            "top_k": top_k
        })
        if len(query.strip()) <= 2:
            results = self._broad_search(query, top_k * 2)
            return results[:top_k]
        expanded_query = self.expand_query(query)
        bm25_top_n = max(top_k * 5, getattr(config, 'BM25_TOP_N', 50))
        bm25_pairs = self._bm25_search(query, bm25_top_n)
        shortlist_indices = [i for i, _ in bm25_pairs]
        q_emb = self._encode_query(expanded_query)
        if shortlist_indices:
            subset = np.array(shortlist_indices, dtype=int)
            sims = cosine_similarity(q_emb, np.array(self.chunk_embeddings)[subset])[0]
            similarities = np.zeros(len(self.chunks), dtype=float)
            similarities[subset] = sims
        else:
            similarities = cosine_similarity(q_emb, np.array(self.chunk_embeddings))[0]
        combined_results = self._combine_search_results(bm25_pairs, similarities, top_k)
        return combined_results

    # --- I18N helpers ---
    def _detect_language(self, text: str) -> str:
        """Lightweight heuristic language detection: returns 'zh-Hant', 'zh-Hans', 'en', etc."""
        if not text:
            return 'zh-Hant'
        # If contains CJK characters
        if re.search(r"[\u4e00-\u9fff]", text):
            # Simple heuristic for Traditional vs Simplified
            trad_chars = '麼應體萬與臺國錄雲這學後開頭時間點號車禮綠碼臺灣臺北於裡'
            if any(ch in text for ch in trad_chars):
                return 'zh-Hant'
            return 'zh-Hans'
        if re.search(r"[\u3040-\u30ff]", text):
            return 'ja'
        if re.search(r"[\uac00-\ud7af]", text):
            return 'ko'
        return 'en'

    def _llm_translate(self, text: str, target_lang: str) -> str:
        """Translate text to target_lang using the same chat API. Falls back to original on error."""
        if not text:
            return text
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        sys = f"You are a professional translator. Translate the user content into {target_lang} preserving meaning and tone. Output only the translated text."
        data = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': sys},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': min(self.max_tokens, 800),
            'temperature': 0.2
        }
        try:
            resp = requests.post(self.openrouter_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            self.log_operation("WARN", "Translation failed, returning original", {"error": str(e)})
            return text

    # --- Generation ---
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], user_lang: Optional[str] = None) -> str:
        # Prepare context with citations
        context_lines = []
        for d in context_docs:
            meta = d.get('metadata', {})
            src = os.path.basename(meta.get('source', 'unknown'))
            idx = meta.get('index', meta.get('chunk_id', 0))
            cs, ce = meta.get('char_start'), meta.get('char_end')
            cite = f"[{src}#{idx}:{cs}-{ce}]" if cs is not None else f"[{src}#{idx}]"
            context_lines.append(f"{d['content']}\n{cite}")
        document_context = "\n\n".join(context_lines)
        conversation_context = self.get_conversation_context()
        system_content_parts = [self.system_prompt]
        system_content_parts.append(f"\n\n相關資料：\n{document_context}")
        if conversation_context:
            system_content_parts.append(f"\n\n對話歷史脈絡：\n{conversation_context}")
        if getattr(config, 'ENABLE_CITATIONS', True):
            system_content_parts.append("\n\n請基於上述資料和對話脈絡提供準確、有幫助的回答，並在引用處以方括號標註來源（例如 [file#chunk:start-end]）。若資訊不足，請明確說明不足之處並提出要澄清的一個問題。")
        else:
            system_content_parts.append("\n\n請基於上述資料和對話脈絡提供準確、有幫助的回答。")
        system_message = "".join(system_content_parts)
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        messages = [{'role': 'system', 'content': system_message}]
        turns = self.conversation_history[-self.conversation_turns_in_prompt:]
        for t in turns:
            if t.get('user'):
                messages.append({'role': 'user', 'content': t['user']})
            if t.get('assistant'):
                messages.append({'role': 'assistant', 'content': t['assistant']})
        messages.append({'role': 'user', 'content': query})
        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            generated_response = result['choices'][0]['message']['content']
            if getattr(config, 'ENABLE_TRANSLATION', True) and user_lang:
                gen_lang = self._detect_language(generated_response)
                if gen_lang != user_lang or getattr(config, 'FORCE_POST_TRANSLATION', True):
                    generated_response = self._llm_translate(generated_response, user_lang)
            token_usage = {}
            if 'usage' in result:
                token_usage = {
                    'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                    'completion_tokens': result['usage'].get('completion_tokens', 0),
                    'total_tokens': result['usage'].get('total_tokens', 0)
                }
            self.log_operation("RESPONSE", "Generated AI response with conversation context", {
                "query": query,
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_length": len(generated_response),
                "context_docs_count": len(context_docs),
                "has_conversation_history": bool(conversation_context),
                "conversation_history_length": len(self.conversation_history),
                "token_usage": token_usage
            })
            return generated_response
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            self.log_operation("ERROR", "Failed to generate response", {
                "query": query,
                "error": str(e),
                "model": self.model_name
            })
            return error_msg

    # --- Chat entrypoint ---
    def chat(self, query: str, top_k: Optional[int] = None) -> str:
        if top_k is None:
            top_k = self.default_top_k
        self.log_operation("CHAT", "Started chat interaction with conversation history", {
            "query": query,
            "top_k": top_k,
            "conversation_history_length": len(self.conversation_history),
            "has_conversation_summary": bool(self.conversation_summary)
        })

        user_lang = self._detect_language(query)
        # Enforce Traditional Chinese for all Chinese queries
        desired_lang = 'zh-Hant' if user_lang in ('zh-Hant', 'zh-Hans') else user_lang
        pivot_lang = getattr(config, 'PIVOT_RETRIEVAL_LANGUAGE', 'zh-Hant')

        retrieval_query = query
        if getattr(config, 'ENABLE_TRANSLATION', True) and user_lang != pivot_lang:
            retrieval_query = self._llm_translate(query, pivot_lang)
            self.log_operation("I18N", "Translated user query for retrieval", {
                "user_lang": user_lang,
                "pivot_lang": pivot_lang,
                "original": query,
                "translated": retrieval_query
            })

        relevant_docs = self.retrieve_relevant_documents(retrieval_query, top_k)
        min_conf = float(getattr(config, 'MIN_CONFIDENCE_FOR_ANSWER', 0.28))
        top_score = float(relevant_docs[0]['similarity']) if relevant_docs else 0.0
        if not relevant_docs or top_score < min_conf:
            mode = str(getattr(config, 'LOW_CONFIDENCE_MODE', 'hedge')).lower()
            if getattr(config, 'CLARIFY_QUESTION_ON_LOW_CONFIDENCE', True) and mode == 'clarify':
                clarify = "我可能無法準確理解您的需求。請問您是想了解哪一方面？例如：功能位置、操作步驟、錯誤訊息或其他？請補充更多關鍵字以便我精準協助。"
                clarify_out = self._llm_translate(clarify, desired_lang)
                self.add_to_conversation_history(query, clarify_out)
                self.log_operation("ANSWER", "Low confidence, asked for clarification", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return clarify_out
            elif mode == 'hedge':
                hint = "我目前能從資料推測的重點如下：\n"
                if relevant_docs:
                    top = relevant_docs[0]
                    ans = top.get('metadata', {}).get('answer') or top.get('content', '')
                    hint += ans[:200] + ("..." if len(ans) > 200 else "")
                follow = "\n\n為了更精準協助，請告訴我您想解決的是哪一類問題（功能位置 / 操作步驟 / 錯誤訊息 / 其他）？"
                hedge = hint + follow
                hedge_out = self._llm_translate(hedge, desired_lang)
                self.add_to_conversation_history(query, hedge_out)
                self.log_operation("ANSWER", "Low confidence, provided hedge + question", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return hedge_out
            else:
                no_results_msg = "抱歉，我無法找到相關的資料來回答您的問題。"
                out = self._llm_translate(no_results_msg, desired_lang)
                self.add_to_conversation_history(query, out)
                self.log_operation("ANSWER", "Low confidence, returned fallback", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return out

        response = self.generate_response(query, relevant_docs, user_lang=desired_lang)
        self.add_to_conversation_history(query, response)
        self.log_operation("ANSWER", "Chat interaction completed with history update", {
            "query": query,
            "response": response,
            "documents_used": len(relevant_docs),
            "top_similarity": float(relevant_docs[0]['similarity']) if relevant_docs else 0.0,
            "conversation_history_length": len(self.conversation_history),
            "has_conversation_summary": bool(self.conversation_summary)
        })
        return response


if __name__ == "__main__":
    print("⚠️  This module is now used as a library.")
    print("🌐 To use the RAG Agent, please run the web interface:")
    print("   python3 app.py")
    print("📚 Or use the startup script:")
    print("   python3 start_web.py")

    def load_txt_data(self, txt_file_path: str):
        """
        Load FAQ data from txt file in Q: A: format
        
        Args:
            txt_file_path: Path to txt file
        """
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by Q: to get each Q&A pair, but keep Q: at the beginning
            lines = content.split('\n')
            current_qa = ""
            qa_count = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    # If we have a previous Q&A pair, process it
                    if current_qa:
                        self._process_qa_pair(current_qa, txt_file_path, qa_count)
                        qa_count += 1
                    # Start new Q&A pair
                    current_qa = line
                elif line and current_qa:
                    # Continue current Q&A pair
                    current_qa += " " + line
            
            # Process the last Q&A pair
            if current_qa:
                self._process_qa_pair(current_qa, txt_file_path, qa_count)
                qa_count += 1
            
            print(f"Loaded {qa_count} Q&A pairs from {txt_file_path}")
            
            # Log the loading operation
            self.log_operation("DATA_LOAD", "Loaded TXT file", {
                "file": txt_file_path,
                "qa_pairs": qa_count
            })
            
        except Exception as e:
            print(f"Error loading TXT file {txt_file_path}: {e}")
            self.log_operation("ERROR", "Failed to load TXT file", {
                "file": txt_file_path,
                "error": str(e)
            })
    
    def _process_qa_pair(self, qa_text: str, source_file: str, index: int):
        """Process a Q&A pair from TXT format"""
        if ' A:' in qa_text:
            question_part, answer_part = qa_text.split(' A:', 1)
            question = question_part.replace('Q:', '').strip()
            answer = answer_part.strip()
            
            if question and answer:
                # Store both question and answer as searchable content
                content = f"問題：{question}\n答案：{answer}"
                self.documents.append(content)
                self.metadata.append({
                    'source': source_file,
                    'index': index,
                    'question': question,
                    'answer': answer
                })
    

    def load_multiple_txt_files(self, txt_files: List[str]):
        """Load multiple TXT files"""
        for txt_file in txt_files:
            if os.path.exists(txt_file):
                self.load_txt_data(txt_file)
            else:
                print(f"File not found: {txt_file}")
    
    def load_multiple_files(self, txt_files: Optional[List[str]] = None):
        """Load multiple TXT files"""
        if txt_files:
            self.load_multiple_txt_files(txt_files)
    
    def _maybe_chunk_documents(self):
        """Split documents into overlapping character chunks for finer retrieval."""
        if self.chunks and self.chunk_metadata:
            return
        enable_chunk = getattr(config, 'ENABLE_CHUNKING', True)
        size = int(getattr(config, 'CHUNK_CHAR_SIZE', 350))
        overlap = int(getattr(config, 'CHUNK_CHAR_OVERLAP', 40))
        if not enable_chunk:
            # One chunk per document
            self.chunks = list(self.documents)
            self.chunk_metadata = list(self.metadata)
            return
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            text = doc
            start = 0
            c_id = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                m = dict(meta)
                m.update({
                    'chunk_id': c_id,
                    'char_start': start,
                    'char_end': end
                })
                metas.append(m)
                if end == len(text):
                    break
                start = end - overlap
                c_id += 1
        self.chunks = chunks
        self.chunk_metadata = metas

    def _build_bm25(self):
        """Build BM25 index for chunks using jieba tokenization."""
        if not getattr(config, 'ENABLE_BM25', True):
            return
        if BM25Okapi is None:
            self._bm25 = None
            return
        if not self.chunks:
            self._maybe_chunk_documents()
        tokenized = [_tokenize_zh(text) for text in self.chunks]
        try:
            self._bm25 = BM25Okapi(tokenized)
        except Exception as e:
            self._bm25 = None
            self.log_operation("WARN", "BM25 build exception", {"error": str(e)})

    def _encode_texts(self, texts: List[str]):
        """Encode texts; if using E5-family, add prompts and normalize."""
        if getattr(self, '_embedding_is_e5', False):
            passages = [f"passage: {t}" for t in texts]
            return self.embedding_model.encode(passages, normalize_embeddings=True)
        return self.embedding_model.encode(texts, normalize_embeddings=True)

    def _encode_query(self, query: str):
        if getattr(self, '_embedding_is_e5', False):
            return self.embedding_model.encode([f"query: {query}"], normalize_embeddings=True)
        return self.embedding_model.encode([query], normalize_embeddings=True)
    
    def create_embeddings(self, save_path: str = "embeddings.pkl"):
        """
        Create embeddings for all documents
        
        Args:
            save_path: Path to save embeddings
        """
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return
        
        print("Creating embeddings for documents...")
        # Build chunks and encode
        self._maybe_chunk_documents()
        # Encode chunk-level embeddings
        self.chunk_embeddings = self._encode_texts(self.chunks)
        # Optional: keep doc-level embeddings for backward compatibility
        try:
            self.embeddings = self._encode_texts(self.documents)
        except Exception:
            self.embeddings = None
        # Build BM25 if enabled
        try:
            self._build_bm25()
        except Exception as e:
            self.log_operation("WARN", "BM25 build failed", {"error": str(e)})
        
        # Save embeddings
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'documents': self.documents,
                'metadata': self.metadata,
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'chunk_embeddings': self.chunk_embeddings
            }, f)
        
        print(f"Embeddings created and saved to {save_path}")
    
    def load_embeddings(self, load_path: str = "embeddings.pkl"):
        """Load pre-computed embeddings"""
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings')
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                # Optional chunk-level fields
                self.chunks = data.get('chunks', []) or list(self.documents)
                self.chunk_metadata = data.get('chunk_metadata', []) or list(self.metadata)
                self.chunk_embeddings = data.get('chunk_embeddings')
                # Rebuild BM25 if enabled
                try:
                    self._build_bm25()
                except Exception as e:
                    self.log_operation("WARN", "BM25 rebuild failed on load", {"error": str(e)})
            print(f"Embeddings loaded from {load_path}")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def add_to_conversation_history(self, user_query: str, assistant_response: str):
        """
        Add a conversation pair to history
        
        Args:
            user_query: User's question
            assistant_response: Assistant's response
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_entry = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": timestamp
        }
        
        self.conversation_history.append(conversation_entry)
        
        # If history is getting too long, create a summary and trim
        if len(self.conversation_history) > self.max_history_length:
            self._manage_conversation_history()
        
        # Log the conversation update
        self.log_operation("CONVERSATION", "Added to conversation history", {
            "history_length": len(self.conversation_history),
            "has_summary": bool(self.conversation_summary)
        })
    
    def _manage_conversation_history(self):
        """
        Manage conversation history by creating summaries when it gets too long
        """
        if len(self.conversation_history) <= self.max_history_length:
            return
        
        # Take the older half of conversations for summarization
        conversations_to_summarize = self.conversation_history[:self.max_history_length//2]
        
        # Create a summary of older conversations
        old_summary = self._create_conversation_summary(conversations_to_summarize)
        
        # Combine with existing summary if any
        if self.conversation_summary:
            combined_summary = f"{self.conversation_summary}\n\n--- 新的對話摘要 ---\n{old_summary}"
        else:
            combined_summary = old_summary
        
        self.conversation_summary = combined_summary
        
        # Keep only the recent conversations
        self.conversation_history = self.conversation_history[self.max_history_length//2:]
        
        self.log_operation("CONVERSATION", "Conversation history summarized and trimmed", {
            "conversations_summarized": len(conversations_to_summarize),
            "remaining_history": len(self.conversation_history),
            "summary_length": len(self.conversation_summary)
        })
    
    def _create_conversation_summary(self, conversations: List[Dict[str, str]]) -> str:
        """
        Create a summary of conversations using the AI model
        
        Args:
            conversations: List of conversation entries to summarize
            
        Returns:
            Summary text
        """
        # Format conversations for summarization
        conversation_text = ""
        for i, conv in enumerate(conversations, 1):
            conversation_text += f"對話 {i}:\n"
            conversation_text += f"用戶: {conv['user']}\n"
            conversation_text += f"助手: {conv['assistant']}\n"
            conversation_text += f"時間: {conv['timestamp']}\n\n"
        
        # Create summarization prompt
        summary_prompt = f"""請為以下對話歷史創建一個簡潔的摘要，重點關注：
1. 用戶主要詢問的問題類型
2. 已經解決的問題
3. 用戶的主要關注點
4. 任何重要的上下文信息

對話歷史：
{conversation_text}

請提供簡潔的摘要："""

        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': summary_prompt
                }
            ],
            'max_tokens': 500,  # Limit summary length
            'temperature': 0.3  # Lower temperature for more consistent summaries
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            summary = result['choices'][0]['message']['content']
            
            # Extract token usage information if available
            token_usage = {}
            if 'usage' in result:
                token_usage = {
                    'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                    'completion_tokens': result['usage'].get('completion_tokens', 0),
                    'total_tokens': result['usage'].get('total_tokens', 0)
                }
            
            self.log_operation("SUMMARY", "Created conversation summary", {
                "conversations_count": len(conversations),
                "summary_length": len(summary),
                "token_usage": token_usage
            })
            
            return summary
            
        except Exception as e:
            # Fallback to simple summary if API fails
            fallback_summary = f"摘要 {len(conversations)} 個對話，時間範圍：{conversations[0]['timestamp']} 至 {conversations[-1]['timestamp']}"
            
            self.log_operation("ERROR", "Failed to create AI summary, using fallback", {
                "error": str(e),
                "fallback_summary": fallback_summary
            })
            
            return fallback_summary
    
    def get_conversation_context(self) -> str:
        """
        Get the current conversation context including summary and recent history
        
        Returns:
            Formatted conversation context
        """
        context_parts = []
        
        # Add summary if exists
        if self.conversation_summary:
            context_parts.append(f"對話摘要：\n{self.conversation_summary}")
        
        # Add recent conversation history
        if self.conversation_history:
            recent_conversations = []
            for conv in self.conversation_history[-5:]:  # Last 5 conversations
                recent_conversations.append(f"用戶: {conv['user']}\n助手: {conv['assistant']}")
            
            if recent_conversations:
                context_parts.append(f"最近對話：\n" + "\n\n".join(recent_conversations))
        
        return "\n\n--- 分隔線 ---\n\n".join(context_parts) if context_parts else ""
    
    def clear_conversation_history(self):
        """Clear all conversation history and summary"""
        self.conversation_history = []
        self.conversation_summary = ""
        
        self.log_operation("CONVERSATION", "Conversation history cleared", {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def expand_query(self, query: str) -> str:
        """
        改进的查询扩展策略 - 更精准，减少噪音
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query
        """
        expanded_terms = set()
        query_lower = query.lower()
        
        # 限制扩展词汇数量
        max_expansion_terms = getattr(config, 'QUERY_EXPANSION_LIMIT', 3)
        
        # 检查是否是否定查询
        is_negation_query = any(neg_word in query_lower for neg_word in 
                               ['找不到', '看不到', '没有', '不见', '无法', '不能', '消失'])
        
        # 优先精确匹配
        for key, terms in self.query_expansions.items():
            if key in query_lower:
                # 只取前N个最相关的扩展词
                expanded_terms.update(terms[:max_expansion_terms])
                self.log_operation("QUERY_EXPANSION", "Precise expansion applied", {
                    "matched_key": key,
                    "expanded_terms": terms[:max_expansion_terms],
                    "query": query
                })
                break  # 找到精确匹配就停止
        
        # 对于否定查询的特殊处理
        if is_negation_query and not expanded_terms:
            for neg_word in ['找不到', '看不到', '没有', '不见']:
                if neg_word in query_lower:
                    after_neg = query_lower.split(neg_word)[-1].strip()
                    if after_neg:
                        for key, terms in self.query_expansions.items():
                            if after_neg in key or key in after_neg:
                                expanded_terms.update(terms[:max_expansion_terms])
                                expanded_terms.add(after_neg)
                                self.log_operation("QUERY_EXPANSION", "Negation-aware expansion", {
                                    "negation_word": neg_word,
                                    "extracted_object": after_neg,
                                    "expanded_terms": terms[:max_expansion_terms]
                                })
                                break
                    break
        
        # 如果还没有扩展且查询很短，进行有限的部分匹配
        if not expanded_terms and len(query.strip()) <= 3:
            for key, terms in self.query_expansions.items():
                if any(char in key for char in query_lower):
                    expanded_terms.update(terms[:2])  # 只取前2个
                    break
        
        if expanded_terms:
            expanded_terms.discard(query_lower)  # 移除原查询避免重复
            expanded_query = query + " " + " ".join(list(expanded_terms)[:max_expansion_terms])
            
            self.log_operation("QUERY_EXPANSION", "Limited expansion completed", {
                "original": query,
                "expanded": expanded_query,
                "terms_added": len(expanded_terms),
                "max_allowed": max_expansion_terms
            })
            return expanded_query
        
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query by removing noise words and negations
        
        Args:
            query: Search query
            
        Returns:
            List of extracted keywords
        """
        # 定义停用词和否定词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '了', '也', '都', '這', '那', '可以', '如何', '什麼', '為什麼', '怎麼', '哪裡'}
        negation_words = {'找不到', '沒有', '不能', '無法', '不會', '看不到', '不見', '消失', '沒看到', '不存在'}
        problem_indicators = {'問題', '錯誤', '故障', '異常', '失敗', '無效'}
        
        query_lower = query.lower()
        
        # 檢查是否包含否定詞，如果有，提取被否定的對象
        keywords = []
        
        # 處理"找不到X"這類查詢，提取X作為關鍵詞
        for neg_word in negation_words:
            if neg_word in query_lower:
                # 找到否定詞後面的內容作為關鍵詞
                parts = query_lower.split(neg_word)
                if len(parts) > 1:
                    after_negation = parts[1].strip()
                    if after_negation:
                        keywords.append(after_negation)
                # 也嘗試找否定詞前面的內容
                before_negation = parts[0].strip()
                if before_negation:
                    keywords.append(before_negation)
        
        # 如果沒有找到否定詞，按常規方式提取關鍵詞
        if not keywords:
            query_terms = query_lower.split()
            for term in query_terms:
                if term not in stop_words and len(term) > 1:
                    keywords.append(term)
        
        # 去重並過濾
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword not in unique_keywords and keyword not in stop_words:
                unique_keywords.append(keyword)
        
        self.log_operation("KEYWORD_EXTRACTION", "Extracted keywords from query", {
            "original_query": query,
            "extracted_keywords": unique_keywords,
            "extraction_method": "negation_aware" if any(neg in query_lower for neg in negation_words) else "standard"
        })
        
        return unique_keywords if unique_keywords else [query.lower()]

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Enhanced keyword-based search with intelligent keyword extraction
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of documents with keyword matches
        """
        keyword_matches = []
        
        # 使用智能關鍵詞提取
        keywords = self._extract_keywords(query)
        
        for idx, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            
            # 計算關鍵詞匹配分數
            exact_matches = 0
            partial_matches = 0
            
            for keyword in keywords:
                if keyword in doc_lower:
                    exact_matches += 1
                else:
                    # 檢查部分匹配（對於較長的關鍵詞）
                    if len(keyword) > 2:
                        keyword_chars = set(keyword)
                        doc_chars = set(doc_lower)
                        if len(keyword_chars.intersection(doc_chars)) / len(keyword_chars) > 0.7:
                            partial_matches += 0.5
            
            total_matches = exact_matches + partial_matches
            
            if total_matches > 0:
                # 計算匹配分數，考慮精確匹配和部分匹配
                keyword_score = total_matches / len(keywords)
                
                # 提升分數如果文檔包含問題相關詞彙
                if any(word in doc_lower for word in ['問題', '如何', '為什麼', '怎麼', '無法', '不能']):
                    keyword_score *= 1.2
                
                keyword_matches.append({
                    'index': idx,
                    'content': doc,
                    'metadata': self.metadata[idx],
                    'keyword_score': keyword_score,
                    'similarity': keyword_score,
                    'matched_keywords': [kw for kw in keywords if kw in doc_lower]
                })
        
        # 按關鍵詞分數排序
        keyword_matches.sort(key=lambda x: x['keyword_score'], reverse=True)
        
        # 記錄搜索結果
        keyword_results_for_log = []
        for match in keyword_matches[:top_k]:
            keyword_results_for_log.append({
                'question': match['metadata']['question'],
                'answer': match['metadata']['answer'][:100] + '...' if len(match['metadata']['answer']) > 100 else match['metadata']['answer'],
                'keyword_score': match['keyword_score'],
                'matched_keywords': match['matched_keywords'],
                'source': match['metadata']['source']
            })
        
        self.log_operation("KEYWORD_SEARCH", "Keyword search completed", {
            "query": query,
            "extracted_keywords": keywords,
            "matches_found": len(keyword_matches),
            "top_scores": [x['keyword_score'] for x in keyword_matches[:3]],
            "results": keyword_results_for_log
        })
        
        return keyword_matches[:top_k]
    
    def _broad_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Broader search for very short queries
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of documents with broad matches
        """
        broad_matches = []
        query_chars = set(query.lower().replace(' ', ''))
        
        for idx, doc in enumerate(self.documents):
            doc_chars = set(doc.lower().replace(' ', ''))
            
            # Calculate character overlap
            if query_chars:
                char_overlap = len(query_chars.intersection(doc_chars)) / len(query_chars)
                
                if char_overlap > 0.3:  # At least 30% character overlap
                    broad_matches.append({
                        'index': idx,
                        'content': doc,
                        'metadata': self.metadata[idx],
                        'broad_score': char_overlap,
                        'similarity': char_overlap  # Use broad score as similarity
                    })
        
        # Sort by broad score
        broad_matches.sort(key=lambda x: x['broad_score'], reverse=True)
        
        # 記錄搜索結果
        broad_results_for_log = []
        for match in broad_matches[:top_k]:
            broad_results_for_log.append({
                'question': match['metadata']['question'],
                'answer': match['metadata']['answer'][:100] + '...' if len(match['metadata']['answer']) > 100 else match['metadata']['answer'],
                'broad_score': match['broad_score'],
                'char_overlap': match['broad_score'],
                'source': match['metadata']['source']
            })
        
        self.log_operation("BROAD_SEARCH", "Broad search completed", {
            "query": query,
            "query_chars": list(query_chars),
            "matches_found": len(broad_matches),
            "top_scores": [x['broad_score'] for x in broad_matches[:3]],
            "results": broad_results_for_log
        })
        
        return broad_matches[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 search over chunks; returns list of (chunk_index, bm25_score)."""
        if not getattr(config, 'ENABLE_BM25', True):
            return []
        if self._bm25 is None:
            try:
                self._build_bm25()
            except Exception as e:
                self.log_operation("WARN", "BM25 not available", {"error": str(e)})
                return []
        if self._bm25 is None:
            return []
        tokens = _tokenize_zh(query)
        scores = self._bm25.get_scores(tokens)
        if top_k >= len(scores):
            idxs = list(range(len(scores)))
        else:
            idxs = np.argpartition(scores, -top_k)[-top_k:]
        ranked = sorted([(int(i), float(scores[int(i)])) for i in idxs], key=lambda x: x[1], reverse=True)
        return ranked
    
    def _combine_search_results(self, bm25_results: List[Tuple[int, float]],
                               semantic_similarities: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Combine BM25 (keyword) and semantic scores at chunk level.
        """
        keyword_weight = getattr(config, 'KEYWORD_WEIGHT', 0.6)
        semantic_weight = getattr(config, 'SEMANTIC_WEIGHT', 0.4)
        semantic_threshold = getattr(config, 'SEMANTIC_THRESHOLD', 0.15)

        # Filter semantic by threshold
        semantic_indices: List[Tuple[int, float]] = [
            (idx, float(score)) for idx, score in enumerate(semantic_similarities) if score >= semantic_threshold
        ]
        semantic_indices.sort(key=lambda x: x[1], reverse=True)

        # Aggregate scores per chunk index
        doc_scores: Dict[int, float] = {}
        # Normalize BM25 to [0,1]
        if bm25_results:
            bm25_max = max(s for _, s in bm25_results) or 1.0
            for idx, s in bm25_results:
                doc_scores[idx] = doc_scores.get(idx, 0.0) + (s / bm25_max) * keyword_weight
        # Add semantic scores
        for idx, s in semantic_indices[:max(top_k * 3, 50)]:
            doc_scores[idx] = doc_scores.get(idx, 0.0) + s * semantic_weight

        # Rank
        combined_results: List[Dict[str, Any]] = []
        for idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
            content = self.chunks[idx] if idx < len(self.chunks) else (self.documents[idx] if idx < len(self.documents) else "")
            combined_results.append({
                'content': content,
                'metadata': meta,
                'similarity': float(score),
                'has_keyword_match': any(i == idx for i, _ in bm25_results),
                'semantic_score': float(semantic_similarities[idx]) if idx < len(semantic_similarities) else 0.0
            })

        # Logging payload
        combined_results_for_log = []
        for r in combined_results:
            m = r.get('metadata', {})
            ans = m.get('answer', '')
            combined_results_for_log.append({
                'question': m.get('question', ''),
                'answer': (ans[:100] + '...') if len(ans) > 100 else ans,
                'combined_score': r['similarity'],
                'semantic_score': r['semantic_score'],
                'has_keyword_match': r['has_keyword_match'],
                'source': m.get('source')
            })

        self.log_operation("COMBINED_SEARCH", "Optimized combined search completed", {
            "keyword_results_count": len(bm25_results),
            "semantic_results_above_threshold": len(semantic_indices),
            "combined_results_count": len(combined_results),
            "keyword_weight": keyword_weight,
            "semantic_weight": semantic_weight,
            "semantic_threshold": semantic_threshold,
            "top_scores": [x['similarity'] for x in combined_results[:3]],
            "results": combined_results_for_log
        })

        return combined_results
    
    def retrieve_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a query with enhanced multi-stage retrieval
        
        Args:
            query: User query
            top_k: Number of top documents to return (uses config value if None)
            
        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = self.default_top_k
        if self.chunk_embeddings is None:
            # Try to build on the fly
            if not self.documents:
                print("No documents loaded. Please load data first.")
                return []
            self._maybe_chunk_documents()
            self.chunk_embeddings = self._encode_texts(self.chunks)
            try:
                self._build_bm25()
            except Exception as e:
                self.log_operation("WARN", "BM25 build failed in retrieval", {"error": str(e)})
        
        # Log the original query
        self.log_operation("SEARCH", "Starting enhanced retrieval", {
            "original_query": query,
            "query_length": len(query.strip()),
            "top_k": top_k
        })
        
    # Step 1: Check if query is very short and needs broad search
        if len(query.strip()) <= 2:
            print("Query very short, using broad character-based search...")
            results = self._broad_search(query, top_k * 2)
            
            # Prepare detailed search results for logging (same format as other search types)
            broad_search_results = []
            for doc in results[:top_k]:
                broad_search_results.append({
                    'similarity_score': float(doc['similarity']),
                    'question': doc['metadata']['question'],
                    'answer': doc['metadata']['answer'],
                    'source': doc['metadata']['source']
                })
            
            self.log_operation("SEARCH", "Broad search completed", {
                "query": query,
                "expanded_query": query,  # No expansion for broad search
                "top_k": top_k,
                "total_documents": len(self.documents),
                "search_type": "broad",
                "keyword_matches": 0,  # No keyword matching in broad search
                "max_semantic_score": float(results[0]['similarity']) if results else 0.0,
                "results": broad_search_results
            })
            return results[:top_k]
        
        # Step 2: Expand query for better semantic matching
        expanded_query = self.expand_query(query)
        
        # Step 3: BM25 over chunks to shortlist
        bm25_top_n = max(top_k * 5, getattr(config, 'BM25_TOP_N', 50))
        bm25_pairs = self._bm25_search(query, bm25_top_n)
        shortlist_indices = [i for i, _ in bm25_pairs]
        
        # Step 4: Semantic similarity over shortlist
        q_emb = self._encode_query(expanded_query)
        if shortlist_indices:
            subset = np.array(shortlist_indices, dtype=int)
            sims = cosine_similarity(q_emb, np.array(self.chunk_embeddings)[subset])[0]
            # Expand to full length array for combiner
            similarities = np.zeros(len(self.chunks), dtype=float)
            similarities[subset] = sims
        else:
            similarities = cosine_similarity(q_emb, np.array(self.chunk_embeddings))[0]
        
        # Step 5: Combine
        combined_results = self._combine_search_results(bm25_pairs, similarities, top_k)
        search_type = "combined"
        
        # Prepare search results for logging
        search_results = []
        for doc in combined_results:
            search_results.append({
                'similarity_score': float(doc['similarity']),
                'question': doc['metadata'].get('question', ''),
                'answer': doc['metadata'].get('answer', ''),
                'source': doc['metadata'].get('source')
            })
        
        # Log the search operation
        self.log_operation("SEARCH", "Enhanced retrieval completed", {
            "query": query,
            "expanded_query": expanded_query,
            "top_k": top_k,
            "total_documents": len(self.documents),
            "search_type": search_type,
            "keyword_matches": len(bm25_pairs),
            "max_semantic_score": float(np.max(similarities)) if len(similarities) else 0.0,
            "results": search_results
        })
        
        return combined_results
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], user_lang: Optional[str] = None) -> str:
        """
        Generate response using OpenRouter with conversation history
        
        Args:
            query: User query
            context_docs: Relevant documents for context
            
        Returns:
            Generated response
        """
        # Prepare context with citations
        context_lines = []
        for d in context_docs:
            meta = d.get('metadata', {})
            src = os.path.basename(meta.get('source', 'unknown'))
            idx = meta.get('index', meta.get('chunk_id', 0))
            cs, ce = meta.get('char_start'), meta.get('char_end')
            cite = f"[{src}#{idx}:{cs}-{ce}]" if cs is not None else f"[{src}#{idx}]"
            context_lines.append(f"{d['content']}\n{cite}")
        document_context = "\n\n".join(context_lines)
        
        # Get conversation history context
        conversation_context = self.get_conversation_context()
        
        # Create system message with context
        system_content_parts = [self.system_prompt]
        
        # Add document context to system message
        system_content_parts.append(f"\n\n相關資料：\n{document_context}")
        
        # Add conversation context if exists
        if conversation_context:
            system_content_parts.append(f"\n\n對話歷史脈絡：\n{conversation_context}")
        
        if getattr(config, 'ENABLE_CITATIONS', True):
            system_content_parts.append("\n\n請基於上述資料和對話脈絡提供準確、有幫助的回答，並在引用處以方括號標註來源（例如 [file#chunk:start-end]）。若資訊不足，請明確說明不足之處並提出要澄清的一個問題。")
        else:
            system_content_parts.append("\n\n請基於上述資料和對話脈絡提供準確、有幫助的回答。")
        
        system_message = "".join(system_content_parts)

        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build role messages with recent conversation turns to improve flow
        messages = [{'role': 'system', 'content': system_message}]
        turns = self.conversation_history[-self.conversation_turns_in_prompt:]
        for t in turns:
            if t.get('user'):
                messages.append({'role': 'user', 'content': t['user']})
            if t.get('assistant'):
                messages.append({'role': 'assistant', 'content': t['assistant']})
        messages.append({'role': 'user', 'content': query})

        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            generated_response = result['choices'][0]['message']['content']
            # Ensure response in user's language if requested
            if getattr(config, 'ENABLE_TRANSLATION', True) and user_lang:
                gen_lang = self._detect_language(generated_response)
                if gen_lang != user_lang or getattr(config, 'FORCE_POST_TRANSLATION', True):
                    generated_response = self._llm_translate(generated_response, user_lang)
            
            # Extract token usage information if available
            token_usage = {}
            if 'usage' in result:
                token_usage = {
                    'prompt_tokens': result['usage'].get('prompt_tokens', 0),
                    'completion_tokens': result['usage'].get('completion_tokens', 0),
                    'total_tokens': result['usage'].get('total_tokens', 0)
                }
            
            # Log the response generation
            self.log_operation("RESPONSE", "Generated AI response with conversation context", {
                "query": query,
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_length": len(generated_response),
                "context_docs_count": len(context_docs),
                "has_conversation_history": bool(conversation_context),
                "conversation_history_length": len(self.conversation_history),
                "token_usage": token_usage
            })
            
            return generated_response
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            self.log_operation("ERROR", "Failed to generate response", {
                "query": query,
                "error": str(e),
                "model": self.model_name
            })
            return error_msg
    
    def chat(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Main chat function that combines retrieval and generation with conversation memory
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Generated response
        """
        if top_k is None:
            top_k = self.default_top_k
        
        # Log the start of chat operation
        self.log_operation("CHAT", "Started chat interaction with conversation history", {
            "query": query,
            "top_k": top_k,
            "conversation_history_length": len(self.conversation_history),
            "has_conversation_summary": bool(self.conversation_summary)
        })
        
        # Language detect and translate query to pivot for retrieval
        user_lang = self._detect_language(query)
        pivot_lang = getattr(config, 'PIVOT_RETRIEVAL_LANGUAGE', 'zh-Hant')
        retrieval_query = query
        if getattr(config, 'ENABLE_TRANSLATION', True) and user_lang != pivot_lang:
            retrieval_query = self._llm_translate(query, pivot_lang)
            self.log_operation("I18N", "Translated user query for retrieval", {
                "user_lang": user_lang,
                "pivot_lang": pivot_lang,
                "original": query,
                "translated": retrieval_query
            })

        # Retrieve relevant documents using translated query if applicable
        relevant_docs = self.retrieve_relevant_documents(retrieval_query, top_k)
        
        # Confidence gating
        min_conf = float(getattr(config, 'MIN_CONFIDENCE_FOR_ANSWER', 0.28))
        top_score = float(relevant_docs[0]['similarity']) if relevant_docs else 0.0
        if not relevant_docs or top_score < min_conf:
            mode = str(getattr(config, 'LOW_CONFIDENCE_MODE', 'hedge')).lower()
            if getattr(config, 'CLARIFY_QUESTION_ON_LOW_CONFIDENCE', True) and mode == 'clarify':
                clarify = "我可能無法準確理解您的需求。請問您是想了解哪一方面？例如：功能位置、操作步驟、錯誤訊息或其他？請補充更多關鍵字以便我精準協助。"
                clarify_out = self._llm_translate(clarify, user_lang)
                self.add_to_conversation_history(query, clarify_out)
                self.log_operation("ANSWER", "Low confidence, asked for clarification", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return clarify_out
            elif mode == 'hedge':
                # Provide best-effort concise answer from top doc + ask 1 clarifying question
                hint = "我目前能從資料推測的重點如下：\n"
                if relevant_docs:
                    top = relevant_docs[0]
                    # extract answer snippet if present
                    ans = top.get('metadata', {}).get('answer') or top.get('content', '')
                    hint += ans[:200] + ("..." if len(ans) > 200 else "")
                follow = "\n\n為了更精準協助，請告訴我您想解決的是哪一類問題（功能位置 / 操作步驟 / 錯誤訊息 / 其他）？"
                hedge = hint + follow
                hedge_out = self._llm_translate(hedge, user_lang)
                self.add_to_conversation_history(query, hedge_out)
                self.log_operation("ANSWER", "Low confidence, provided hedge + question", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return hedge_out
            else:  # strict
                no_results_msg = "抱歉，我無法找到相關的資料來回答您的問題。"
                out = self._llm_translate(no_results_msg, user_lang)
                self.add_to_conversation_history(query, out)
                self.log_operation("ANSWER", "Low confidence, returned fallback", {
                    "query": query,
                    "top_similarity": top_score,
                    "min_required": min_conf
                })
                return out
        
        # Generate response with conversation context; ensure output in user's language
        response = self.generate_response(query, relevant_docs, user_lang=user_lang)
        
        # Add this conversation to history
        self.add_to_conversation_history(query, response)
        
        # Log the final answer
        self.log_operation("ANSWER", "Chat interaction completed with history update", {
            "query": query,
            "response": response,
            "documents_used": len(relevant_docs),
            "top_similarity": float(relevant_docs[0]['similarity']) if relevant_docs else 0.0,
            "conversation_history_length": len(self.conversation_history),
            "has_conversation_summary": bool(self.conversation_summary)
        })
        
        return response

if __name__ == "__main__":
    print("⚠️  This module is now used as a library.")
    print("🌐 To use the RAG Agent, please run the web interface:")
    print("   python3 app.py")
    print("📚 Or use the startup script:")
    print("   python3 start_web.py")
