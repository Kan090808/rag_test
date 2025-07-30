from flask import Flask, render_template, request, jsonify, session
import json
from datetime import datetime
import os
import sys
import socket

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent
import config

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this to a secure secret key

# Global RAG agent instance
rag_agent = None

def find_available_port(start_port=5001, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def kill_process_on_port(port):
    """Kill process using the specified port (macOS/Linux)"""
    try:
        import subprocess
        # Find process using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"🔄 Killing process {pid} using port {port}...")
                subprocess.run(['kill', '-9', pid], capture_output=True)
            return True
    except Exception as e:
        print(f"⚠️  Could not kill process on port {port}: {e}")
    return False

def initialize_rag_agent():
    """Initialize the RAG agent with configuration"""
    global rag_agent
    try:
        # Initialize RAG agent
        rag_agent = RAGAgent(
            openrouter_api_key=config.OPENROUTER_API_KEY,
            model_name=config.MODEL_NAME,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            log_file=config.LOG_FILE,
            system_prompt=config.SYSTEM_PROMPT
        )
        
        # Load documents
        print("Loading documents...")
        
        # Load TXT files if any
        if hasattr(config, 'TXT_FILES') and config.TXT_FILES:
            for txt_file in config.TXT_FILES:
                if os.path.exists(txt_file):
                    rag_agent.load_txt_data(txt_file)
                    print(f"Loaded TXT: {txt_file}")
        
        # Try to load existing embeddings first
        if os.path.exists(config.EMBEDDINGS_FILE):
            try:
                rag_agent.load_embeddings(config.EMBEDDINGS_FILE)
                print(f"Loaded existing embeddings from {config.EMBEDDINGS_FILE}")
            except Exception as e:
                print(f"Could not load existing embeddings: {e}")
                # Create new embeddings if loading fails
                if rag_agent.documents:
                    rag_agent.create_embeddings(config.EMBEDDINGS_FILE)
                    print("Created new embeddings")
        else:
            # Create new embeddings
            if rag_agent.documents:
                rag_agent.create_embeddings(config.EMBEDDINGS_FILE)
                print("Created new embeddings")
        
        if rag_agent.documents:
            print("RAG Agent initialized successfully!")
            return True
        else:
            print("Warning: No documents loaded!")
            return False
            
    except Exception as e:
        print(f"Error initializing RAG Agent: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        if not rag_agent:
            return jsonify({'error': 'RAG Agent not initialized'}), 500
        
        # Handle special commands
        if user_message.lower() == 'clear_history':
            rag_agent.clear_conversation_history()
            return jsonify({
                'response': '對話歷史已清除！',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if user_message.lower() == 'show_history':
            if rag_agent.conversation_history or rag_agent.conversation_summary:
                context = rag_agent.get_conversation_context()
                return jsonify({
                    'response': f'當前對話歷史：\n{context}',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                return jsonify({
                    'response': '目前沒有對話歷史。',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Get response from RAG agent
        response = rag_agent.chat(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/status')
def status():
    """Check system status"""
    try:
        status_info = {
            'rag_agent_initialized': rag_agent is not None,
            'model_name': config.MODEL_NAME if rag_agent else None,
            'documents_loaded': len(rag_agent.documents) if rag_agent else 0,
            'conversation_history_length': len(rag_agent.conversation_history) if rag_agent else 0,
        }
        return jsonify(status_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        if rag_agent:
            rag_agent.clear_conversation_history()
            return jsonify({'message': 'History cleared successfully'})
        else:
            return jsonify({'error': 'RAG Agent not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    
    # Check if user wants command-line mode
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Command-line interface mode
        print("Starting RAG Agent in Command-Line Mode...")
        print("Initializing RAG Agent...")
        
        if initialize_rag_agent():
            print("="*50)
            print("RAG Agent Ready! Ask me anything!")
            print("新功能：我現在能記住我們整個對話過程！")
            print("Type 'exit' or 'quit' to end the conversation.")
            print("Type 'clear_history' to clear conversation history.")
            print("Type 'show_history' to view conversation history.")
            print("="*50 + "\n")
            
            # Chat loop
            while True:
                try:
                    user_query = input("You: ").strip()
                    
                    if user_query.lower() in ['exit', 'quit', '退出']:
                        print("Agent: 再見！感謝使用RAG Agent！")
                        break
                    
                    if user_query.lower() == 'clear_history':
                        rag_agent.clear_conversation_history()
                        print("Agent: 對話歷史已清除！")
                        continue
                        
                    if user_query.lower() == 'show_history':
                        if rag_agent.conversation_history or rag_agent.conversation_summary:
                            context = rag_agent.get_conversation_context()
                            print(f"\nAgent: 當前對話歷史：\n{context}\n")
                        else:
                            print("Agent: 目前沒有對話歷史。\n")
                        continue
                    
                    if not user_query:
                        continue
                    
                    print("\nAgent: 正在搜尋相關資料並生成回答...")
                    response = rag_agent.chat(user_query)
                    print(f"Agent: {response}\n")
                    
                except KeyboardInterrupt:
                    print("\n\nAgent: 再見！感謝使用RAG Agent！")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        else:
            print("Failed to initialize RAG Agent. Please check your configuration and data files.")
    else:
        # Web interface mode (default)
        print("Starting RAG Agent Web Interface...")
        print("Initializing RAG Agent...")
        
        if initialize_rag_agent():
            # Find an available port
            preferred_port = 5001
            available_port = find_available_port(preferred_port)
            
            if available_port is None:
                print(f"⚠️  No available ports found in range {preferred_port}-{preferred_port + 9}")
                print("🔄 Attempting to free up port 5001...")
                if kill_process_on_port(preferred_port):
                    available_port = preferred_port
                else:
                    print("❌ Could not find or free up an available port!")
                    exit(1)
            
            if available_port != preferred_port:
                print(f"⚠️  Port {preferred_port} is in use, using port {available_port} instead")
            
            print("="*50)
            print("RAG Agent Web Interface Ready!")
            print(f"Access the web interface at: http://localhost:{available_port}")
            print("="*50)
            app.run(debug=False, host='0.0.0.0', port=available_port)
        else:
            print("Failed to initialize RAG Agent. Please check your configuration and data files.")
