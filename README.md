# DocMind AI — Document Intelligence & RAG Chatbot

DocMind AI is an AI-powered document intelligence application built with Streamlit, Python, ChromaDB, and Groq. It allows users to upload documents, search their content, and ask questions using Retrieval-Augmented Generation (RAG).

The application extracts and preprocesses document text, converts it into numerical embeddings, stores it in a vector database, and uses a large language model to generate context-aware answers.

## Live Demo

Try the deployed application here:

**[Launch DocMind AI](https://docmind-ai-guchthpjwdrfslzifazwyd.streamlit.app)**

Explore document uploads, semantic search, and AI-powered question answering using Retrieval-Augmented Generation (RAG).

## Features

- **Multi-format document support:** Process documents, spreadsheets, presentations, images, emails, and notebooks.
- **OCR support:** Extract text from supported images and scanned PDF pages using Tesseract OCR.
- **Document preprocessing:** Clean and normalize extracted text.
- **Text chunking:** Split document text into overlapping word-based chunks.
- **Semantic search:** Retrieve relevant text using Sentence Transformers embeddings and ChromaDB.
- **RAG-based question answering:** Generate answers using retrieved document context.
- **Source references:** Display filenames and chunk information associated with retrieved results.
- **Document catalog:** Maintain document-level records to support document discovery and summary requests.
- **Conversation history:** Use previous messages to help interpret follow-up questions.
- **Long-term user memory:** Store selected user information in a separate collection when supported by the application.
- **Multiple upload options:** Upload individual files, select a folder, or upload a ZIP archive.
- **Session-specific data isolation:** Use separate document, memory, and catalog collections for each Streamlit session.
- **Session-specific file storage:** Store uploaded files in a directory associated with the current session.
- **Session-scoped data clearing:** Clear the current session's data without intentionally clearing every visitor's collections or uploaded files.

## Supported File Formats

The application supports the following extensions, subject to the availability of required dependencies and external tools.

| Category | Formats |
|---|---|
| Documents | PDF, DOC, DOCX, ODT, RTF |
| Text and markup | TXT, MD, HTML, HTM, XML |
| Spreadsheets and data | CSV, TSV, JSON, JSONL, XLS, XLSX, XLSM, ODS |
| Presentations | PPT, PPTX, ODP |
| Images | PNG, JPG, JPEG, TIF, TIFF, BMP, WEBP |
| Email | EML, MSG |
| E-books | EPUB |
| Notebooks | IPYNB |

Some formats require additional packages or software. For example, legacy Office formats may require LibreOffice, `.xls` files require `xlrd`, `.ods` files require `odfpy`, and `.msg` files require `extract-msg`.

Actual format support depends on the installed dependencies and the extraction functions available in the current application.

## Technologies Used

- **Python** — Application logic and document processing
- **Streamlit** — Web interface
- **Groq API** — Large language model access
- **OpenAI GPT-OSS 20B** (`openai/gpt-oss-20b`) — Configured language model
- **Sentence Transformers** — Text embedding generation
- **all-MiniLM-L6-v2** — Embedding model
- **ChromaDB** — Vector database
- **Tesseract OCR** — Text extraction from supported images and scanned pages
- **PyMuPDF** — PDF processing
- **Pandas and NumPy** — Data processing

## How It Works

1. **Upload:** The user uploads supported documents through the Streamlit interface.
2. **Extraction:** The application extracts text using format-specific processing functions.
3. **Preprocessing:** Extracted text is cleaned and normalized.
4. **Chunking:** Text is divided into smaller, overlapping word-based chunks.
5. **Embedding:** Sentence Transformers converts chunks into numerical vectors.
6. **Storage:** Chunks, embeddings, and metadata are stored in ChromaDB collections.
7. **Retrieval:** The application searches for relevant document chunks when a question is asked.
8. **Answer generation:** Retrieved context is sent to Groq's configured language model.
9. **Response:** The application displays the generated answer and available source details.

## Text Chunking Configuration

The application uses the following settings, as configured in the current implementation:

- **Chunk size:** `500` words
- **Chunk overlap:** `50` words
- **Step size:** `450` words

The overlap helps preserve context between consecutive chunks.

## Database Configuration

DocMind AI uses **local persistent ChromaDB storage** rather than Chroma Cloud.

The database path is configured in the application:

```python
CHROMA_PATH = "./chroma_db"
```

### Session-specific collections

The application creates separate collections for each Streamlit session:

- `documents_<session_suffix>` — Stores document chunks and associated metadata.
- `memory_<session_suffix>` — Stores session-specific memory.
- `catalog_<session_suffix>` — Stores document-level information used for discovery and summaries.

The application creates a unique session identifier and uses part of that identifier to distinguish the collections belonging to each session.

This design helps prevent one visitor's session cleanup from unintentionally deleting another visitor's collections.

### Uploaded document storage

Uploaded files are stored in a session-specific directory under the following local data location:

```text
AppData/Local/DocMindAI/uploaded_documents/<session_id>/
```

The actual path is constructed from the current user's home directory. Each session has its own upload directory, which helps keep file cleanup scoped to that session.

### Storage limitations

Local persistent storage does not guarantee permanent storage on a hosted platform. On Streamlit Community Cloud, local database contents and uploaded files should not be assumed to survive instance replacement, storage resets, or redeployments.

The application does not currently configure an external persistent vector database.

Back up important local data and use an appropriately configured persistent storage solution if long-term data retention is required.

## Project Structure

```text
DocMind-AI/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `app.py` — Main application, document processing, embeddings, retrieval, and question answering.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Excludes configured local secrets, virtual environments, databases, and uploaded documents.
- `README.md` — Project documentation.

Additional files or directories may exist in the development environment.

## Prerequisites

Before running the application, install:

- Python
- Git
- A Groq API key
- Python packages listed in `requirements.txt`
- Tesseract OCR for image and scanned-page text extraction
- LibreOffice if required for legacy Office format conversion

Additional Python packages may be required for specific document formats.

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/mohammedhussainali1346/DocMind-AI.git
cd DocMind-AI
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv docmind_aivenv
.\docmind_aivenv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate activation method for your terminal or run the environment's Python executable directly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your actual Groq API key.

The application loads the key from `.env` and checks Streamlit secrets if the environment variable is unavailable.

**Security:** Never commit your `.env` file or publish your API key. Keep `.env` and `.streamlit/secrets.toml` excluded from Git tracking.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use DocMind AI.

## Deployment on Streamlit Community Cloud

### Step 1: Push the project to GitHub

Ensure your repository includes the files required by the application, such as:

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`

After making changes, commit and push them to GitHub:

```bash
git add README.md
git commit -m "Update README documentation"
git push origin main
```

Run these commands only when you have new changes to commit.

### Step 2: Configure the Streamlit app

1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Create a new app, or open your existing deployed app.
4. Select the `mohammedhussainali1346/DocMind-AI` repository.
5. Select the appropriate branch.
6. Set the main file path to `app.py`.
7. Deploy the application if it has not already been deployed.

### Step 3: Configure Streamlit secrets

Open the deployed app's settings and locate the Secrets section.

Add your Groq API key in TOML format:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Use your actual key in the private secrets configuration. Do not place the key in `app.py`, `README.md`, or any public repository file.

The current application does not require Chroma Cloud credentials.

### Step 4: Test the deployment

After deployment:

1. Open the application URL.
2. Confirm that the app starts successfully.
3. Upload a small supported document.
4. Ask a question about its contents.
5. Check that an answer and available source details appear.
6. Test the session-scoped data-clearing feature.
7. Review the deployment logs if an error occurs.

**Deployment limitation:** The current version uses local ChromaDB storage. It does not configure an external persistent vector database, so local database contents and uploaded files may be lost when the cloud instance is replaced or its storage is reset.

## Troubleshooting

| Problem | What to check |
|---|---|
| Missing Groq API key | Check `.env` locally or the Streamlit Cloud Secrets configuration. |
| Dependency installation fails | Review `requirements.txt` and the deployment logs. |
| OCR fails | Confirm that Tesseract OCR is installed and accessible. |
| Legacy Office conversion fails | Install LibreOffice and make sure its executable is available. |
| A document cannot be processed | Verify the extension, required package, and document contents. |
| No relevant chunks are retrieved | Check extraction, chunk creation, embeddings, and retrieval settings. |
| ChromaDB errors | Check database initialization and local storage permissions. |
| Data disappears after redeployment | The current application stores ChromaDB data locally; use a persistent external database if durable cloud storage is required. |
| Session data is not isolated as expected | Check session identifier initialization, collection names, upload paths, and cleanup logic. |

## Security and Privacy

- Never publish API keys or other credentials.
- Keep `.env` and `.streamlit/secrets.toml` out of version control.
- Avoid uploading confidential documents unless you understand how they are processed and stored.
- Back up important local database data before making changes.
- Remember that `.gitignore` does not remove files that Git is already tracking.
- Session-specific collections and upload directories help separate visitor data, but they should not be treated as a complete security guarantee.
- Use appropriate authentication, authorization, and persistent storage controls if the application is used for sensitive or production data.

## Future Improvements

Possible improvements include:

- Persistent cloud database integration
- Better document retrieval and reranking
- More robust handling of scanned documents and tables
- Expanded source citations
- Document deletion and lifecycle management
- Improved error handling and automated testing
- Stronger authentication and access controls for production deployments

## Author

**Muhammad Hussain Ali**

GitHub: [mohammedhussainali1346](https://github.com/mohammedhussainali1346)

## License

No license has been specified in the current repository. Add a `LICENSE` file if you want to define how others may use, modify, and distribute this project.