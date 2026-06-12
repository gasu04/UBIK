# Google Drive RAG with DeepSeek R1 14B

Complete RAG (Retrieval Augmented Generation) system that loads documents from Google Drive and answers questions using DeepSeek R1 14B with local embeddings.

## Features

✓ **Multi-format support**: PDF, DOCX, TXT, MD files
✓ **Google Drive integration**: Direct access to your cloud documents
✓ **Local embeddings**: nomic-embed-text (privacy + no API costs)
✓ **Vector database**: ChromaDB for fast retrieval
✓ **Powerful AI**: DeepSeek R1 14B reasoning model
✓ **Interactive queries**: Ask questions in real-time

## Quick Start

### 1. Ensure Ollama is running

```bash
# In a separate terminal
ollama serve &
```

### 2. Run the script

```bash
cd "/Volumes/990PRO 4T/DeepSeek"
source venv/bin/activate
python gdrive_rag_deepseek.py
```

### 3. First run authentication

- Browser will open for Google Drive authorization
- Allow access to read your files
- Token saved to `token.json` for future use

### 4. Ask questions!

```
❓ Your question: What are the main topics in my documents?
```

## Configuration

### Process specific folder

Edit line 244 in `gdrive_rag_deepseek.py`:

```python
folder_id = "1ABC123xyz_yourFolderID"  # Get ID from Google Drive URL
```

### Adjust file types

```python
file_types=['pdf', 'docx', 'txt', 'md']  # Remove types you don't need
```

### Change chunk size

```python
chunk_size=1000,      # Larger = more context, fewer chunks
chunk_overlap=200     # Overlap between chunks
```

### Retrieve more documents

```python
k=3  # Number of relevant docs to retrieve (line 214)
```

## File Structure

```
/Volumes/990PRO 4T/DeepSeek/
├── gdrive_rag_deepseek.py     # Main script
├── credentials.json            # Google API credentials (your existing file)
├── token.json                  # Auth token (auto-generated)
├── gdrive_downloads/           # Downloaded files cache
└── chroma_gdrive_db/          # Vector database (persistent)
```

## How It Works

```
┌─────────────────┐
│  Google Drive   │
│  (Your Docs)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Download &    │
│  Parse Docs     │
│ (PDF/DOCX/TXT)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ nomic-embed-text│  ← Local embeddings
│   (Ollama)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │  ← Vector storage
│ (./chroma_db)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Your Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Top-K  │  ← Find relevant docs
│   Documents     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DeepSeek R1    │  ← Generate answer
│      14B        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Your Answer    │
└─────────────────┘
```

## Tips

### Reprocess documents

Delete the vector database and run again:
```bash
rm -rf ./chroma_gdrive_db
python gdrive_rag_deepseek.py
```

### Monitor performance

- Watch for download speeds
- Large PDFs may take time to process
- First query loads the model (30s)
- Subsequent queries are faster

### Optimize for speed

1. Use smaller `chunk_size` (500 instead of 1000)
2. Reduce `k` value (2 instead of 3)
3. Filter to specific file types only

### Optimize for accuracy

1. Use larger `chunk_size` (1500-2000)
2. Increase `k` value (4-5)
3. Increase `chunk_overlap` (300-400)

## Troubleshooting

### "Ollama server not responding"
```bash
ollama serve &
```

### "Model not found"
```bash
ollama pull deepseek-r1:14b
ollama pull nomic-embed-text
```

### "credentials.json not found"
- Download from Google Cloud Console
- Place in `/Volumes/990PRO 4T/DeepSeek/`

### "Out of memory"
- Close other applications
- Reduce chunk_size
- Process fewer files at once

## Advanced Usage

### Add web page support

Install additional library:
```bash
pip install playwright
playwright install chromium
```

Then use `WebBaseLoader` in the script.

### Custom prompts

Modify the QA chain to use custom prompts:
```python
from langchain.prompts import PromptTemplate

template = """Use the following context to answer the question.
If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
```

## Cost Analysis

Everything runs locally:
- ✓ **DeepSeek R1 14B**: Free (local)
- ✓ **nomic-embed-text**: Free (local)
- ✓ **ChromaDB**: Free (local)
- ✓ **Storage**: ~500MB for 100 documents

**Total cost: $0/month** 💰

Compare to cloud RAG:
- OpenAI GPT-4: ~$30-100/month
- OpenAI Embeddings: ~$5-20/month
- Pinecone/Vector DB: ~$70-140/month

**You save: $105-260/month** 🎉
