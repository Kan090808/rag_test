#!/usr/bin/env python3
"""
Setup script for RAG Agent
This script will install all required dependencies and set up the RAG agent.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_files():
    """Check if CSV and TXT files exist"""
    try:
        import config
        csv_files = getattr(config, 'CSV_FILES', [])
        txt_files = getattr(config, 'TXT_FILES', [])
    except ImportError:
        print("⚠️  config.py not found. Please copy config_template.py to config.py and configure your settings.")
        return [], []
    
    existing_csv_files = []
    existing_txt_files = []
    
    print("Checking CSV files...")
    for file in csv_files:
        if os.path.exists(file):
            existing_csv_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"⚠️  Not found: {file}")
    
    print("\nChecking TXT files...")
    for file in txt_files:
        if os.path.exists(file):
            existing_txt_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"⚠️  Not found: {file}")
    
    return existing_csv_files, existing_txt_files

def main():
    print("="*50)
    print("RAG Agent Setup")
    print("="*50)
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n" + "="*50)
    print("Checking data files...")
    print("="*50)
    
    # Check CSV and TXT files
    existing_csv_files, existing_txt_files = check_data_files()
    
    total_files = len(existing_csv_files) + len(existing_txt_files)
    if total_files > 0:
        print(f"\n✅ Found {len(existing_csv_files)} CSV file(s) and {len(existing_txt_files)} TXT file(s) for RAG knowledge base.")
        print(f"📊 Total data files: {total_files}")
    else:
        print("\n⚠️  No data files found. Please add your CSV and/or TXT files to the current directory.")
        print("   Or update the CSV_FILES and TXT_FILES arrays in config.py")
    
    print("\n" + "="*50)
    print("Setup Instructions:")
    print("="*50)
    print("1. Get your OpenRouter API key from https://openrouter.ai/")
    print("2. Copy config_template.py to config.py and add your API key")
    print("3. Update CSV_FILES and TXT_FILES arrays in config.py if needed")
    print("4. Run: python rag_agent.py")
    print("="*50)
    
    # Remove embeddings.pkl after setup
    embeddings_path = "embeddings.pkl"
    if os.path.exists(embeddings_path):
        try:
            os.remove(embeddings_path)
            print(f"🗑️  Removed {embeddings_path} after setup.")
        except Exception as e:
            print(f"⚠️  Could not remove {embeddings_path}: {e}")

if __name__ == "__main__":
    main()
