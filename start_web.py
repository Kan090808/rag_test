#!/usr/bin/env python3
"""
Web Interface Startup Script for RAG Agent
This script will set up and start the web interface for the RAG agent.
"""

import subprocess
import sys
import os
import socket

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

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_config():
    """Check if config.py exists"""
    if not os.path.exists("config.py"):
        print("⚠️  config.py not found!")
        print("Please copy config_template.py to config.py and configure your settings.")
        return False
    
    try:
        import config
        if not hasattr(config, 'OPENROUTER_API_KEY') or not config.OPENROUTER_API_KEY:
            print("⚠️  Please set your OPENROUTER_API_KEY in config.py")
            return False
        print("✅ Configuration file found and valid!")
        return True
    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    try:
        import config
        txt_files = getattr(config, 'TXT_FILES', [])
    except ImportError:
        print("⚠️  Could not import config")
        return False
    
    if not txt_files:
        print("⚠️  No TXT files configured in config.py")
        return False
    
    existing_files = []
    print("📁 Checking data files...")
    for file in txt_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"⚠️  Not found: {file}")
    
    if not existing_files:
        print("❌ No data files found! Please ensure your TXT files exist.")
        return False
    
    return True

def start_web_interface():
    """Start the Flask web interface"""
    print("\n" + "="*50)
    print("🚀 Starting RAG Agent Web Interface...")
    print("="*50)
    
    try:
        # Import and run the Flask app
        from app import app, initialize_rag_agent
        
        print("🤖 Initializing RAG Agent...")
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
                    return False
            
            if available_port != preferred_port:
                print(f"⚠️  Port {preferred_port} is in use, using port {available_port} instead")
            
            print("\n" + "="*50)
            print("🌐 RAG Agent Web Interface is Ready!")
            print(f"📍 Access the interface at: http://localhost:{available_port}")
            print("💡 Press Ctrl+C to stop the server")
            print("="*50 + "\n")
            
            # Start Flask app
            app.run(debug=False, host='0.0.0.0', port=available_port)
        else:
            print("❌ Failed to initialize RAG Agent!")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 RAG Agent Web Interface stopped by user.")
        return True
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False

def main():
    """Main function"""
    print("🤖 RAG Agent Web Interface Setup")
    print("="*40)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your Python environment.")
        return
    
    # Step 2: Check configuration
    if not check_config():
        print("❌ Configuration check failed. Please fix your config.py file.")
        return
    
    # Step 3: Check data files
    if not check_data_files():
        print("❌ Data files check failed. Please ensure your data files exist.")
        return
    
    # Step 4: Start web interface
    start_web_interface()

if __name__ == "__main__":
    main()
