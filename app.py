from flask import Flask, render_template, request, jsonify, session
import json
from datetime import datetime
import os
import sys
import socket
import uuid

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent
import config

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this to a secure secret key

# Global RAG agent instance
rag_agent = None

# Store session data for questions and their logs
question_sessions = {}

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
        
        # Generate unique question ID
        question_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Handle special commands
        if user_message.lower() == 'clear_history':
            rag_agent.clear_conversation_history()
            response_data = {
                'response': '對話歷史已清除！',
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'question_id': question_id
            }
            
            # Store basic session info even for commands
            question_sessions[question_id] = {
                'question': user_message,
                'response': response_data['response'],
                'timestamp': start_time,
                'logs': []
            }
            
            return jsonify(response_data)
        
        if user_message.lower() == 'show_history':
            if rag_agent.conversation_history or rag_agent.conversation_summary:
                context = rag_agent.get_conversation_context()
                response_text = f'當前對話歷史：\n{context}'
            else:
                response_text = '目前沒有對話歷史。'
            
            response_data = {
                'response': response_text,
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'question_id': question_id
            }
            
            # Store basic session info even for commands
            question_sessions[question_id] = {
                'question': user_message,
                'response': response_data['response'],
                'timestamp': start_time,
                'logs': []
            }
            
            return jsonify(response_data)
        
        # Store initial log state
        initial_log_size = 0
        if os.path.exists(config.LOG_FILE):
            try:
                with open(config.LOG_FILE, 'r', encoding='utf-8') as f:
                    initial_log_size = len(f.read())
            except:
                pass
        
        # Get response from RAG agent
        response = rag_agent.chat(user_message)
        end_time = datetime.now()
        
        # Capture logs generated for this question
        logs = []
        if os.path.exists(config.LOG_FILE):
            try:
                with open(config.LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    new_content = content[initial_log_size:]
                    if new_content.strip():
                        # Parse JSON log entries more carefully
                        log_entries = []
                        
                        # Split by the log separator line
                        log_blocks = new_content.split('-' * 80)
                        
                        for block in log_blocks:
                            block = block.strip()
                            if block.startswith('{') and block.endswith('}'):
                                try:
                                    log_entry = json.loads(block)
                                    log_entries.append(log_entry)
                                except json.JSONDecodeError as e:
                                    print(f"Failed to parse log entry: {e}")
                                    # Try to extract JSON from multi-line block
                                    lines = block.split('\n')
                                    json_lines = []
                                    in_json = False
                                    brace_count = 0
                                    
                                    for line in lines:
                                        if line.strip().startswith('{'):
                                            in_json = True
                                            brace_count = line.count('{') - line.count('}')
                                            json_lines = [line]
                                        elif in_json:
                                            json_lines.append(line)
                                            brace_count += line.count('{') - line.count('}')
                                            if brace_count <= 0:
                                                try:
                                                    json_str = '\n'.join(json_lines)
                                                    log_entry = json.loads(json_str)
                                                    log_entries.append(log_entry)
                                                    break
                                                except:
                                                    pass
                        
                        logs = log_entries
            except Exception as e:
                print(f"Error reading logs: {e}")
                logs = [{"error": f"Failed to read logs: {str(e)}"}]
        
        # Store session data
        question_sessions[question_id] = {
            'question': user_message,
            'response': response,
            'timestamp': start_time,
            'end_time': end_time,
            'logs': logs,
            'processing_time': (end_time - start_time).total_seconds()
        }
        
        return jsonify({
            'response': response,
            'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'question_id': question_id
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

@app.route('/logs/<question_id>')
def get_logs(question_id):
    """Get logs for a specific question"""
    try:
        if question_id not in question_sessions:
            return jsonify({'error': 'Question not found'}), 404
        
        session_data = question_sessions[question_id]
        return jsonify({
            'question_id': question_id,
            'question': session_data['question'],
            'response': session_data['response'],
            'timestamp': session_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': session_data.get('processing_time', 0),
            'logs': session_data['logs']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/change_model', methods=['POST'])
def change_model():
    """Change the model used by RAG agent"""
    try:
        data = request.get_json()
        new_model = data.get('model_name', '').strip()
        
        if not new_model:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Validate model name
        allowed_models = [
            'anthropic/claude-3.5-haiku',
            'openai/gpt-4.1',
            'openai/gpt-4.1-mini'
        ]
        
        if new_model not in allowed_models:
            return jsonify({'error': f'Model not allowed. Available models: {", ".join(allowed_models)}'}), 400
        
        if not rag_agent:
            return jsonify({'error': 'RAG Agent not initialized'}), 500
        
        # Change the model
        success = rag_agent.change_model(new_model)
        
        if success:
            # Update the config module's MODEL_NAME for consistency
            config.MODEL_NAME = new_model
            
            return jsonify({
                'message': f'Model changed to {new_model} successfully',
                'new_model': new_model
            })
        else:
            return jsonify({'error': 'Failed to change model'}), 500
            
    except Exception as e:
        print(f"Error in change_model endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

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
