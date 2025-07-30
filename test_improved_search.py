#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试改进后的RAG搜索功能
测试查询："找不到时刻"
"""

import os
import sys
from rag_agent import RAGAgent
import config

def test_improved_search():
    """测试改进后的搜索功能"""
    
    print("=== 测试改进后的RAG搜索功能 ===\n")
    
    # 初始化RAG Agent
    try:
        agent = RAGAgent(
            openrouter_api_key=config.OPENROUTER_API_KEY,
            model_name=config.MODEL_NAME,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            log_file="test_search.log"
        )
        print("✅ RAG Agent 初始化成功")
    except Exception as e:
        print(f"❌ RAG Agent 初始化失败: {e}")
        return
    
    # 加载数据
    try:
        if os.path.exists("0708.txt"):
            agent.load_txt_data("0708.txt")
            print(f"✅ 数据加载成功，共 {len(agent.documents)} 个文档")
        else:
            print("❌ 找不到数据文件 0708.txt")
            return
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 创建或加载嵌入
    try:
        if not agent.load_embeddings("embeddings.pkl"):
            print("创建新的嵌入...")
            agent.create_embeddings("embeddings.pkl")
        print("✅ 嵌入准备就绪")
    except Exception as e:
        print(f"❌ 嵌入处理失败: {e}")
        return
    
    # 测试查询列表
    test_queries = [
        "找不到時刻",
        "找不到时刻",
        "看不到時刻",
        "沒有時刻功能",
        "時刻不見了",
        "時刻",
        "時刻功能",
        "為什麼沒有時刻",
        "時刻在哪裡"
    ]
    
    print("\n=== 开始测试查询 ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 测试查询 {i}: '{query}'")
        print("-" * 50)
        
        try:
            # 测试关键词提取
            keywords = agent._extract_keywords(query)
            print(f"📝 提取的关键词: {keywords}")
            
            # 测试查询扩展
            expanded_query = agent.expand_query(query)
            print(f"🔄 扩展后的查询: '{expanded_query}'")
            
            # 执行搜索
            results = agent.retrieve_relevant_documents(query, top_k=3)
            
            if results:
                print(f"✅ 找到 {len(results)} 个相关结果:")
                for j, result in enumerate(results, 1):
                    print(f"\n结果 {j} (相似度: {result['similarity']:.3f}):")
                    print(f"问题: {result['metadata']['question']}")
                    print(f"答案: {result['metadata']['answer']}")
                    if 'matched_keywords' in result:
                        print(f"匹配关键词: {result.get('matched_keywords', [])}")
            else:
                print("❌ 没有找到相关结果")
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"❌ 查询 '{query}' 处理失败: {e}")
            print("\n" + "="*80 + "\n")
    
    # 测试完整对话
    print("🤖 测试完整对话功能:")
    print("-" * 50)
    
    test_chat_query = "找不到時刻"
    try:
        response = agent.chat(test_chat_query)
        print(f"用户: {test_chat_query}")
        print(f"助手: {response}")
    except Exception as e:
        print(f"❌ 对话测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print(f"详细日志请查看: test_search.log")

def main():
    """主函数"""
    test_improved_search()

if __name__ == "__main__":
    main()
