import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import List, Dict, Any, Optional
import pickle
import logging
from datetime import datetime

class RAGAgent:
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-3.5-haiku", 
                 max_tokens: int = 1000, temperature: float = 0.7, log_file: str = "log.txt",
                 system_prompt: str = "你是一個專業的客服助手。請根據以下提供的資料來回答用戶的問題。"):
        """
        Initialize the RAG Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model_name: Model to use (default: Claude 3.5 Haiku)
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            log_file: Path to log file for tracking operations
            system_prompt: System prompt to guide the AI's responses
        """
        self.openrouter_api_key = openrouter_api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.default_top_k = 3  # Default value
        self.log_file = log_file
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Setup logging
        self._setup_logging()
        
        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.log_operation("SYSTEM", "Loading sentence transformer model", {})
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for documents and embeddings
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Conversation history management
        self.conversation_history = []  # List of {"user": str, "assistant": str, "timestamp": str}
        self.max_history_length = 20   # Maximum number of conversations to keep
        self.conversation_summary = ""  # Summary of older conversations
        
        # Query expansion configuration for Letstalk app
        self.query_expansions = {
            # "群": ["群組", "群聊", "群組聊天", "建立群組", "群組功能", "群組成員"],
            # "好友": ["好友", "朋友", "聯絡人", "加好友", "好友列表", "好友管理"],
            # "訊息": ["訊息", "消息", "聊天", "對話", "傳送", "接收"],
            # "登入": ["登入", "登錄", "登入問題", "無法登入", "登入失敗"],
            # "帳號": ["帳號", "帳戶", "用戶", "註冊", "帳號管理", "帳號設定"],
            # "檔案": ["檔案", "文件", "附件", "圖片", "照片", "影片"],
            # "設定": ["設定", "設置", "配置", "選項", "偏好設定"],
            # "通知": ["通知", "提醒", "推播", "通知設定", "消息提醒"],
            # "封鎖": ["封鎖", "阻擋", "黑名單", "封鎖好友", "解除封鎖"],
            # "下載": ["下載", "載入", "保存", "儲存", "無法下載"],
            # "刪除": ["刪除", "移除", "清除", "刪掉", "消除"],
            # "密碼": ["密碼", "密碼重設", "忘記密碼", "修改密碼", "密碼問題"],
            # "聊天": ["聊天", "對話", "談話", "交談", "聊天室"],
            # "功能": ["功能", "特色", "選項", "工具", "服務"],
            # "問題": ["問題", "錯誤", "故障", "異常", "無法使用"],
            "安裝": ["安裝 letstalk", "letstalk 安裝檔", "letstalk 安裝包", "更新 letstalk", "letstalk 更新檔"], 
        }
        
        print("RAG Agent initialized successfully!")
        self.log_operation("SYSTEM", "RAG Agent initialized", {
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create log file if it doesn't exist
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
        self.embeddings = self.embedding_model.encode(self.documents)
        
        # Save embeddings
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        print(f"Embeddings created and saved to {save_path}")
    
    def load_embeddings(self, load_path: str = "embeddings.pkl"):
        """Load pre-computed embeddings"""
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.documents = data['documents']
                self.metadata = data['metadata']
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
            
            self.log_operation("SUMMARY", "Created conversation summary", {
                "conversations_count": len(conversations),
                "summary_length": len(summary)
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
        Expand short queries using synonyms and related terms
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query
        """
        expanded_terms = set()
        query_lower = query.lower()
        
        # Find matching expansion terms
        for key, terms in self.query_expansions.items():
            if key in query_lower:
                expanded_terms.update(terms)
        
        # Also add partial matches for short queries
        if len(query.strip()) <= 3:
            for key, terms in self.query_expansions.items():
                if any(char in key for char in query_lower):
                    expanded_terms.update(terms[:3])  # Add first 3 synonyms for partial matches
        
        if expanded_terms:
            # Remove the original query terms to avoid duplication
            expanded_terms.discard(query_lower)
            expanded_query = query + " " + " ".join(expanded_terms)
            
            self.log_operation("QUERY_EXPANSION", "Query expanded", {
                "original": query,
                "expanded": expanded_query,
                "added_terms": list(expanded_terms)
            })
            return expanded_query
        
        return query
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search for exact matches
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of documents with keyword matches
        """
        keyword_matches = []
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        for idx, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            
            # Calculate keyword match score
            exact_matches = sum(1 for term in query_terms if term in doc_lower)
            if exact_matches > 0:
                # Higher score for more matches
                keyword_score = exact_matches / len(query_terms)
                keyword_matches.append({
                    'index': idx,
                    'content': doc,
                    'metadata': self.metadata[idx],
                    'keyword_score': keyword_score,
                    'similarity': keyword_score  # Use keyword score as similarity
                })
        
        # Sort by keyword score
        keyword_matches.sort(key=lambda x: x['keyword_score'], reverse=True)
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
        return broad_matches[:top_k]
    
    def _combine_search_results(self, keyword_results: List[Dict[str, Any]], 
                               semantic_similarities: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Combine keyword and semantic search results
        
        Args:
            keyword_results: Results from keyword search
            semantic_similarities: Similarity scores from semantic search
            top_k: Number of top results to return
            
        Returns:
            Combined and ranked results
        """
        # Get semantic search results
        top_semantic_indices = np.argsort(semantic_similarities)[::-1][:top_k * 2]
        
        # Create a score dictionary for all documents
        doc_scores = {}
        
        # Add keyword scores (weight: 0.4)
        for result in keyword_results:
            idx = result['index']
            doc_scores[idx] = doc_scores.get(idx, 0) + result['keyword_score'] * 0.4
        
        # Add semantic scores (weight: 0.6)
        for idx in top_semantic_indices:
            semantic_score = semantic_similarities[idx]
            doc_scores[idx] = doc_scores.get(idx, 0) + semantic_score * 0.6
        
        # Create combined results
        combined_results = []
        for idx, combined_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            combined_results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': combined_score
            })
        
        return combined_results
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a query with enhanced multi-stage retrieval
        
        Args:
            query: User query
            top_k: Number of top documents to return
            
        Returns:
            List of relevant documents with metadata
        """
        if self.embeddings is None:
            print("No embeddings available. Please create or load embeddings first.")
            return []
        
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
            
            self.log_operation("SEARCH", "Broad search completed", {
                "query": query,
                "results_count": len(results),
                "search_type": "broad"
            })
            return results[:top_k]
        
        # Step 2: Expand query for better semantic matching
        expanded_query = self.expand_query(query)
        
        # Step 3: Perform keyword search
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Step 4: Perform semantic search with expanded query
        query_embedding = self.embedding_model.encode([expanded_query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Step 5: If we have keyword matches and good semantic scores, combine them
        if keyword_results and np.max(similarities) > 0.3:
            combined_results = self._combine_search_results(keyword_results, similarities, top_k)
            search_type = "combined"
        # Step 6: If keyword search found good matches, prioritize them
        elif keyword_results and len(keyword_results) >= top_k:
            combined_results = keyword_results[:top_k]
            search_type = "keyword_priority"
        # Step 7: Otherwise, use pure semantic search
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            combined_results = []
            for idx in top_indices:
                combined_results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': similarities[idx]
                })
            search_type = "semantic_only"
        
        # Prepare search results for logging
        search_results = []
        for doc in combined_results:
            search_results.append({
                'similarity_score': float(doc['similarity']),
                'question': doc['metadata']['question'],
                'answer': doc['metadata']['answer'],
                'source': doc['metadata']['source']
            })
        
        # Log the search operation
        self.log_operation("SEARCH", "Enhanced retrieval completed", {
            "query": query,
            "expanded_query": expanded_query,
            "top_k": top_k,
            "total_documents": len(self.documents),
            "search_type": search_type,
            "keyword_matches": len(keyword_results),
            "max_semantic_score": float(np.max(similarities)),
            "results": search_results
        })
        
        return combined_results
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate response using OpenRouter with conversation history
        
        Args:
            query: User query
            context_docs: Relevant documents for context
            
        Returns:
            Generated response
        """
        # Prepare context from retrieved documents
        document_context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Get conversation history context
        conversation_context = self.get_conversation_context()
        
        # Create enhanced prompt with conversation history
        prompt_parts = [self.system_prompt]
        
        # Add conversation context if exists
        if conversation_context:
            prompt_parts.append(f"\n對話歷史脈絡：\n{conversation_context}")
        
        # Add document context
        prompt_parts.append(f"\n相關資料：\n{document_context}")
        
        # Add current query
        prompt_parts.append(f"\n用戶問題：{query}")
        
        prompt_parts.append("\n請基於上述脈絡和資料提供準確、有幫助的回答。如果當前問題與之前的對話相關，請結合對話歷史給出更好的回答：")
        
        prompt = "".join(prompt_parts)

        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            generated_response = result['choices'][0]['message']['content']
            
            # Log the response generation
            self.log_operation("RESPONSE", "Generated AI response with conversation context", {
                "query": query,
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_length": len(generated_response),
                "context_docs_count": len(context_docs),
                "has_conversation_history": bool(conversation_context),
                "conversation_history_length": len(self.conversation_history)
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
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query, top_k)
        
        if not relevant_docs:
            no_results_msg = "抱歉，我無法找到相關的資料來回答您的問題。"
            
            # Still add to conversation history even if no results
            self.add_to_conversation_history(query, no_results_msg)
            
            self.log_operation("ANSWER", "No relevant documents found", {
                "query": query,
                "response": no_results_msg
            })
            return no_results_msg
        
        # Generate response with conversation context
        response = self.generate_response(query, relevant_docs)
        
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
