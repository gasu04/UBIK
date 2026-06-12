# Google Drive RAG System Setup Guide

This guide will help you set up a RAG (Retrieval-Augmented Generation) system that uses documents from your Google Drive with DeepSeek AI.

## 📋 Prerequisites

- **Python 3.10+** installed
- **Ollama** installed and running
- **DeepSeek model** downloaded (`ollama pull deepseek-r1:14b`)
- A **Google account** with access to Google Drive
- Documents in a **Google Drive folder**

---

## 🚀 Quick Start

### Step 1: Set Up Virtual Environment

```bash
# Navigate to your project directory
cd "/Volumes/990PRO 4T/DeepSeek"

# Activate your existing virtual environment
source venv/bin/activate

# Or create a new one if needed
# python3 -m venv venv
# source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements_google_drive.txt
```

This installs:
- LangChain packages (core, community, Ollama integration)
- Google Drive API libraries
- Document loaders (PDF, DOCX, CSV, etc.)
- ChromaDB for vector storage
- Sentence transformers for embeddings

---

## 🔐 Step 3: Set Up Google Drive API Credentials

### 3.1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Select a project"** → **"New Project"**
3. Enter a project name (e.g., "RAG System")
4. Click **"Create"**

### 3.2: Enable Google Drive API

1. In the Cloud Console, go to **"APIs & Services"** → **"Library"**
2. Search for **"Google Drive API"**
3. Click on it, then click **"Enable"**

### 3.3: Create OAuth 2.0 Credentials

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"Create Credentials"** → **"OAuth client ID"**
3. If prompted, configure the OAuth consent screen:
   - Choose **"External"** (unless you have a Google Workspace)
   - Fill in the required fields:
     - App name: "RAG System"
     - User support email: your email
     - Developer contact: your email
   - Click **"Save and Continue"**
   - Skip "Scopes" (click **"Save and Continue"**)
   - Add yourself as a test user
   - Click **"Save and Continue"**

4. Back at **"Create OAuth client ID"**:
   - Application type: **"Desktop app"**
   - Name: "RAG Desktop Client"
   - Click **"Create"**

5. Download the credentials:
   - Click the **download icon** next to your new OAuth 2.0 Client ID
   - Save the file as **`credentials.json`** in your project directory

**Important:** Keep `credentials.json` secure and never share it publicly!

---

## 📂 Step 4: Get Your Google Drive Folder ID

1. Open **Google Drive** in your browser
2. Navigate to the folder containing your documents
3. Look at the URL in your browser's address bar:
   ```
   https://drive.google.com/drive/folders/1a2B3c4D5e6F7g8H9i0J
                                           ^^^^^^^^^^^^^^^^^^^^
                                           This is your Folder ID
   ```
4. Copy the **Folder ID** (the part after `/folders/`)

---

## 📥 Step 5: Download Documents from Google Drive

Run the download script:

```bash
python download_from_drive.py
```

**What happens:**
1. You'll be prompted to enter your **Google Drive Folder ID**
2. Choose a download location (default: `./google_drive_docs`)
3. Choose whether to download subfolders recursively
4. **First time only:** A browser window will open asking you to:
   - Sign in to your Google account
   - Grant permission to access your Google Drive
5. The script will download all files from the folder

**Supported file types:**
- Text files (`.txt`)
- Markdown (`.md`)
- PDFs (`.pdf`)
- CSV files (`.csv`)
- Google Docs (exported as `.txt`)
- Google Sheets (exported as `.csv`)
- Google Slides (exported as `.pdf`)

**Files created:**
- `token.json` - Your authentication token (auto-generated, keep secure)

---

## 🤖 Step 6: Run the RAG System

```bash
python rag_with_docs.py
```

**What happens:**
1. You'll be prompted for:
   - **Documents folder path** (default: `./google_drive_docs`)
   - **Vector store location** (default: `./chroma_db`)

2. The system will:
   - Load all documents from the folder
   - Create embeddings using sentence-transformers
   - Build a vector database with ChromaDB
   - Connect to your DeepSeek model via Ollama
   - Start an interactive Q&A session

3. Ask questions about your documents!

**Example interaction:**
```
❓ Question: What are the main topics in these documents?

🤔 Thinking...

💡 Answer: Based on the documents, the main topics include...
```

---

## 📝 Usage Tips

### Updating Documents

To add new documents from Google Drive:

1. Add files to your Google Drive folder
2. Run `python download_from_drive.py` again
3. Run `python rag_with_docs.py`
4. When prompted about the existing vector store, choose **'n'** to rebuild it with new documents

### Customizing the System

Edit `rag_with_docs.py` to customize:

```python
# At the top of the file:
DEFAULT_MODEL = "deepseek-r1:14b"  # Change AI model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Change embedding model

# In create_vector_store():
chunk_size=1000,  # Adjust chunk size (smaller = more precise, larger = more context)
chunk_overlap=200,  # Adjust overlap

# In create_rag_chain():
search_kwargs={"k": 3}  # Number of document chunks to retrieve (3-5 recommended)
temperature=0.7  # AI creativity (0.0 = focused, 1.0 = creative)
```

### File Types

By default, the system supports:
- ✅ Text files (`.txt`, `.md`)
- ✅ CSV files (`.csv`)
- ✅ PDFs (`.pdf`) - if `pypdf` installed
- ✅ Word documents (`.docx`) - if `python-docx` installed

To add more file types, install additional packages:
```bash
pip install unstructured  # Handles many formats
```

---

## 🔧 Troubleshooting

### "credentials.json not found"
- Make sure you downloaded the OAuth credentials from Google Cloud Console
- Save it as `credentials.json` in the same directory as the scripts

### "ModuleNotFoundError: No module named 'google'"
```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

### "Error connecting to Ollama"
1. Check if Ollama is running:
   ```bash
   ollama serve
   ```
2. Verify DeepSeek model is installed:
   ```bash
   ollama list
   ollama pull deepseek-r1:14b  # If not installed
   ```

### "No documents found"
- Check that files were downloaded to the correct directory
- Verify the path when running `rag_with_docs.py`
- Check that files have supported extensions (`.txt`, `.pdf`, `.csv`, `.md`)

### Documents not loading
```bash
# For PDFs:
pip install pypdf

# For Word documents:
pip install python-docx unstructured

# For all document types:
pip install unstructured
```

### "Token has expired"
- Delete `token.json`
- Run `python download_from_drive.py` again
- Re-authenticate in the browser

---

## 🔒 Security Notes

**Keep these files private:**
- `credentials.json` - Your Google API credentials
- `token.json` - Your authentication token
- `chroma_db/` - Your vector database (contains document content)

**Best practices:**
- Add these to `.gitignore` if using version control
- Don't share these files publicly
- Regenerate credentials if accidentally exposed

---

## 📚 Project Structure

After setup, your directory should look like this:

```
DeepSeek/
├── venv/                          # Virtual environment
├── credentials.json               # Google API credentials (keep private!)
├── token.json                     # Auth token (auto-generated, keep private!)
├── download_from_drive.py         # Script to download from Google Drive
├── rag_with_docs.py              # Main RAG system script
├── requirements_google_drive.txt  # Python dependencies
├── SETUP_GOOGLE_DRIVE_RAG.md     # This guide
├── google_drive_docs/            # Downloaded documents
│   ├── document1.txt
│   ├── document2.pdf
│   └── ...
└── chroma_db/                    # Vector database (auto-generated)
    └── ...
```

---

## 🎯 Next Steps

Once everything is working:

1. **Add more documents** to your Google Drive folder
2. **Experiment with questions** to see what works best
3. **Adjust chunk sizes** for better results
4. **Try different models** (e.g., `deepseek-r1:7b` for faster responses)
5. **Build a web interface** using Streamlit or Gradio

---

## 💡 Example Questions to Ask

Depending on your documents, try:

- "What are the main topics discussed in these documents?"
- "Summarize the key points about [topic]"
- "What does the document say about [specific subject]?"
- "Compare the information about X and Y"
- "Find all mentions of [keyword]"

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Review the Troubleshooting section above
3. Make sure all prerequisites are met
4. Check that Ollama is running: `ollama list`

---

## ✅ Verification Checklist

Before running the system, verify:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip install -r requirements_google_drive.txt`)
- [ ] Ollama installed and running
- [ ] DeepSeek model downloaded (`ollama pull deepseek-r1:14b`)
- [ ] Google Cloud project created
- [ ] Google Drive API enabled
- [ ] OAuth credentials downloaded as `credentials.json`
- [ ] Google Drive Folder ID copied
- [ ] Documents downloaded successfully

---

**Happy querying! 🚀**
