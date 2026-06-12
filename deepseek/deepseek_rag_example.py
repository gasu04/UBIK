#!/usr/bin/env python3
"""
DeepSeek RAG Example with LangChain, ChromaDB, and Local Embeddings

This script demonstrates how to:
1. Use local Ollama embeddings (nomic-embed-text)
2. Store documents in ChromaDB vector database
3. Query using DeepSeek R1 14B model
4. Build a complete RAG pipeline
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

def main():
    print("🚀 Setting up DeepSeek RAG with local embeddings...\n")

    # 1. Initialize local embeddings (nomic-embed-text from Ollama)
    print("📊 Loading local embeddings model...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    print("✓ Embeddings model loaded\n")

    # 2. Initialize DeepSeek model
    print("🤖 Loading DeepSeek R1 14B model...")
    llm = Ollama(
        model="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0.7
    )
    print("✓ DeepSeek model loaded\n")

    # 3. Sample documents (replace with your own data)
    print("📚 Creating sample documents...")
    documents = [
        Document(
            page_content="DeepSeek is a powerful AI model focused on reasoning capabilities.",
            metadata={"source": "doc1", "topic": "AI"}
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) combines retrieval with generation.",
            metadata={"source": "doc2", "topic": "RAG"}
        ),
        Document(
            page_content="ChromaDB is an open-source vector database designed for AI applications.",
            metadata={"source": "doc3", "topic": "Database"}
        ),
        Document(
            page_content="Local embeddings provide privacy and eliminate API costs.",
            metadata={"source": "doc4", "topic": "Embeddings"}
        ),
    ]

    # 4. Split documents (useful for larger texts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} document chunks\n")

    # 5. Create ChromaDB vector store
    print("💾 Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Data persists here
    )
    print("✓ Vector store created\n")

    # 6. Create retrieval chain
    print("🔗 Creating RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  # Retrieve top 2 documents
        ),
        return_source_documents=True
    )
    print("✓ RAG chain ready\n")

    # 7. Query the system
    print("=" * 60)
    print("💬 Asking questions to DeepSeek RAG system...")
    print("=" * 60 + "\n")

    questions = [
        "What is RAG and how does it work?",
        "Why should I use local embeddings?"
    ]

    for question in questions:
        print(f"❓ Question: {question}\n")
        result = qa_chain.invoke({"query": question})
        print(f"🤖 Answer: {result['result']}\n")
        print(f"📄 Sources used: {[doc.metadata['source'] for doc in result['source_documents']]}\n")
        print("-" * 60 + "\n")

    print("✅ Demo complete!\n")
    print("📝 Next steps:")
    print("  1. Replace sample documents with your own data")
    print("  2. Adjust chunk_size and chunk_overlap for your use case")
    print("  3. Experiment with different k values for retrieval")
    print("  4. Try different prompting strategies\n")

if __name__ == "__main__":
    main()
