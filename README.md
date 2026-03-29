# 🤖 Autonomous RAG Assistant with Web Fallback

An intelligent document-chat application built with Python, LangChain, and Streamlit. This agent allows users to upload multiple PDFs and ask questions. 

Unlike standard RAG pipelines, this is a **fully autonomous agent**. It reads the retrieved documents and independently decides if it has enough information to answer the user's question. If the documents are insufficient, it autonomously triggers a DuckDuckGo web search to find the missing context.

## ✨ Features
* **Agentic Reasoning:** Powered by LangChain's `AgentExecutor` and Llama 3.3.
* **Transparent 'Chain of Thought':** The UI exposes the agent's internal monologue, showing exactly *why* it decided to search the web or stick to local documents.
* **Local Vector Database:** Uses ChromaDB and HuggingFace embeddings for fast, accurate document retrieval.
* **Contextual Memory:** Maintains chat history for seamless, multi-turn conversations.

## 🚀 Try it out
https://autonomous-rag-agent.streamlit.app