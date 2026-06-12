#!/usr/bin/env python3
"""
RAG System with Google Drive Documents
Loads documents from a local folder and creates a RAG system using DeepSeek + ChromaDB
"""

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    CSVLoader
)

# Try to import optional loaders
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Note: PyPDF not installed. PDF files will be skipped.")
    print("Install with: pip install pypdf")

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Note: python-docx not installed. DOCX files will be skipped.")
    print("Install with: pip install python-docx unstructured")

# Configuration
DEFAULT_DOCS_PATH = "./google_drive_docs"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_documents_from_directory(directory_path):
    """
    Load all supported documents from a directory.

    Supports: .txt, .md, .csv, .pdf (if pypdf installed), .docx (if python-docx installed)
    """
    if not os.path.exists(directory_path):
        print(f"\n❌ Error: Directory not found: {directory_path}")
        return []

    documents = []
    supported_extensions = ['.txt', '.md', '.csv']
    if PDF_AVAILABLE:
        supported_extensions.append('.pdf')
    if DOCX_AVAILABLE:
        supported_extensions.append('.docx')

    print(f"\nScanning directory: {directory_path}")
    print(f"Supported file types: {', '.join(supported_extensions)}")
    print("-" * 50)

    file_count = 0
    error_count = 0

    # Walk through directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()

            # Skip hidden files and unsupported types
            if file.startswith('.') or file_ext not in supported_extensions:
                continue

            try:
                print(f"📄 Loading: {file}")

                # Choose appropriate loader based on file type
                if file_ext in ['.txt', '.md']:
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext == '.csv':
                    loader = CSVLoader(file_path)
                elif file_ext == '.pdf' and PDF_AVAILABLE:
                    loader = PyPDFLoader(file_path)
                elif file_ext == '.docx' and DOCX_AVAILABLE:
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:
                    print(f"   ⏭️  Skipped (loader not available)")
                    continue

                docs = loader.load()
                documents.extend(docs)
                file_count += 1
                print(f"   ✅ Loaded ({len(docs)} pages/sections)")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                error_count += 1

    print("-" * 50)
    print(f"✅ Successfully loaded: {file_count} files")
    if error_count > 0:
        print(f"❌ Errors: {error_count} files")
    print(f"📊 Total document sections: {len(documents)}")

    return documents

def create_vector_store(documents, persist_directory, embedding_model_name):
    """Create or load vector store from documents"""

    # Initialize embeddings
    print(f"\n📥 Loading embeddings model: {embedding_model_name}")
    print("   (First time will download ~90MB)")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Check if vector store already exists
    persist_path = os.path.expanduser(persist_directory)

    if os.path.exists(persist_path) and os.listdir(persist_path):
        print(f"\n📚 Loading existing vector store from: {persist_path}")

        use_existing = input("Use existing vector store? (Y/n): ").strip().lower()
        if use_existing != 'n':
            vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=embeddings
            )
            print("✅ Vector store loaded!")
            return vectorstore
        else:
            print("Creating new vector store...")

    # Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=200,  # Overlap helps maintain context
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Create vector store
    print(f"\n💾 Creating vector store...")
    print("   (This may take a few minutes for large document sets)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    print(f"✅ Vector store created and saved to: {persist_path}")
    return vectorstore

def create_rag_chain(vectorstore, model_name):
    """Create the RAG chain with retriever and LLM"""

    print(f"\n🤖 Connecting to Ollama model: {model_name}")
    print("   (Make sure Ollama is running)")

    try:
        llm = OllamaLLM(model=model_name, temperature=0.7)

        # Test connection
        print("   Testing connection...")
        test_response = llm.invoke("Hi")
        print("   ✅ Connection successful!")

    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("\nMake sure:")
        print("1. Ollama is running")
        print("2. The model is installed: ollama pull deepseek-r1:14b")
        exit(1)

    # Create retriever
    print("\n🔍 Setting up retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )

    # Create prompt template
    template = """You are a helpful assistant answering questions based on the provided context.

Context from documents:
{context}

Question: {question}

Answer the question based on the context above. If you cannot answer based on the context, say so. Be detailed and helpful."""

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain using LCEL
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready!")
    return chain

def interactive_query(chain):
    """Run interactive query loop"""
    print("\n" + "=" * 70)
    print("🚀 RAG SYSTEM READY")
    print("=" * 70)
    print("\nAsk questions about your documents!")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("=" * 70)

    while True:
        try:
            print("\n" + "-" * 70)
            question = input("\n❓ Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break

            print("\n🤔 Thinking...\n")
            result = chain.invoke(question)
            print(f"💡 Answer:\n{result}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    print("=" * 70)
    print("RAG SYSTEM WITH GOOGLE DRIVE DOCUMENTS")
    print("=" * 70)

    # Get configuration
    docs_path = input(f"\nPath to documents folder (default: {DEFAULT_DOCS_PATH}): ").strip()
    if not docs_path:
        docs_path = DEFAULT_DOCS_PATH

    persist_dir = input(f"Vector store location (default: {DEFAULT_PERSIST_DIR}): ").strip()
    if not persist_dir:
        persist_dir = DEFAULT_PERSIST_DIR

    # Load documents
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 70)
    documents = load_documents_from_directory(docs_path)

    if not documents:
        print("\n❌ No documents found. Please check the path and try again.")
        print(f"   Looking in: {os.path.abspath(docs_path)}")
        exit(1)

    # Create vector store
    print("\n" + "=" * 70)
    print("STEP 2: CREATING VECTOR STORE")
    print("=" * 70)
    vectorstore = create_vector_store(documents, persist_dir, DEFAULT_EMBEDDING_MODEL)

    # Create RAG chain
    print("\n" + "=" * 70)
    print("STEP 3: SETTING UP RAG CHAIN")
    print("=" * 70)
    chain = create_rag_chain(vectorstore, DEFAULT_MODEL)

    # Interactive query
    interactive_query(chain)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
