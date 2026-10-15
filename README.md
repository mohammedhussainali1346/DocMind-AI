# DocMind AI — Document Intelligence & RAG Chatbot

DocMind AI is an AI-powered document intelligence application built with Streamlit, Python, ChromaDB, and Groq. It allows users to upload documents, search their content, and ask questions using Retrieval-Augmented Generation (RAG).

The application extracts and preprocesses document text, converts it into numerical embeddings, stores it in a local vector database, and uses a large language model to generate context-aware answers.

## Features

- **Multi-format document support:** Process documents, spreadsheets, presentations, images, emails, and notebooks.
- **OCR support:** Extract text from supported images and scanned PDF pages using Tesseract OCR.
- **Document preprocessing:** Clean and normalize extracted text.
- **Text chunking:** Split document text into overlapping word-based chunks.
- **Semantic search:** Retrieve relevant text using Sentence Transformers embeddings and ChromaDB.
- **RAG-based question answering:** Generate answers using retrieved document context.
- **Source references:** Display filenames and chunk information associated with retrieved results.
- **Document catalog:** Keep document-level records to support document discovery and summary requests.
- **Conversation history:** Use previous messages to help interpret follow-up questions.
- **Long-term user memory:** Optionally store relevant, stable user information in a separate collection.
- **Multiple upload options:** Upload individual files, select a folder, or upload a ZIP archive.

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

## Technologies Used

- **Python** — Application logic and document processing
- **Streamlit** — Web interface
- **Groq API** — Large language model access
- **OpenAI GPT-OSS 20B** (`openai/gpt-oss-20b`) — Configured language model
- **Sentence Transformers** — Text embedding generation
- **all-MiniLM-L6-v2** — Embedding model
- **ChromaDB** — Local vector database
- **Tesseract OCR** — Text extraction from supported images and scanned pages
- **PyMuPDF** — PDF processing
- **Pandas and NumPy** — Data processing

## How It Works

1. **Upload:** The user uploads supported documents through the Streamlit interface.
2. **Extraction:** The application extracts text using format-specific processing functions.
3. **Preprocessing:** Extracted text is cleaned and normalized.
4. **Chunking:** Text is divided into smaller, overlapping word-based chunks.
5. **Embedding:** Sentence Transformers converts chunks into numerical vectors.
6. **Storage:** Chunks, embeddings, and metadata are stored in local ChromaDB collections.
7. **Retrieval:** The application searches for relevant document chunks when a question is asked.
8. **Answer generation:** The retrieved context is sent to Groq's configured language model.
9. **Response:** The application displays the generated answer and available source details.

## Text Chunking Configuration

The current application uses these settings:

- Chunk size: `500` words
- Chunk overlap: `50` words
- Step size: `450` words

The overlap helps preserve context between consecutive chunks.

## Database Configuration

DocMind AI currently uses **local persistent ChromaDB**, not Chroma Cloud.

The application initializes the database with:

```python
CHROMA_PATH = "./chroma_db"
```

It creates three collections:

- `document_intelligence` — Stores document chunks and their metadata.
- `user_memory` — Stores selected long-term user memories.
- `document_catalog` — Stores document-level information for discovery and summaries.

The database is created locally in the `chroma_db/` directory. Uploaded files are stored in the `uploaded_documents/` directory during processing.

**Important:** Local database files and uploaded documents are excluded from Git using `.gitignore`. On Streamlit Community Cloud, local files should not be assumed to survive app restarts, redeployments, or instance replacement. Back up any important data before changing or rebuilding the environment.

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
- `.gitignore` — Excludes local secrets, virtual environments, databases, and uploaded documents.
- `README.md` — Project documentation.

## Prerequisites

Before running the application, install:

- Python
- Git
- A Groq API key
- Python packages listed in `requirements.txt`
- Tesseract OCR for image and scanned-page text extraction
- LibreOffice if you need the legacy Office formats handled through conversion

Additional Python packages are listed in `requirements.txt` for specific formats.

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

Ensure your repository includes:

- `app.py`
- `requirements.txt`
- `.gitignore`
- `README.md`

Commit and push the README:

```bash
git add README.md
git commit -m "Add project README"
git push
```

### Step 2: Create the Streamlit app

1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Create a new app.
4. Select the `mohammedhussainali1346/DocMind-AI` repository.
5. Select the branch containing your project.
6. Set the main file path to `app.py`.
7. Deploy the application.

### Step 3: Configure Streamlit secrets

Open your deployed app's settings and locate the Secrets section.

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
6. Review the deployment logs if an error occurs.

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

## Security and Privacy

- Never publish API keys or other credentials.
- Keep `.env` and `.streamlit/secrets.toml` out of version control.
- Avoid uploading confidential documents unless you understand how they are processed and stored.
- Back up important local database data before making changes.
- Remember that `.gitignore` does not remove files that Git is already tracking.

## Future Improvements

Possible improvements include:

- Persistent cloud database integration
- Better document retrieval and reranking
- More robust handling of scanned documents and tables
- Expanded source citations
- Document deletion and lifecycle management
- Improved error handling and automated testing

## Author

**Muhammad Hussain Ali**

GitHub: [mohammedhussainali1346](https://github.com/mohammedhussainali1346)

## License

No license has been specified in the current repository. Add a `LICENSE` file if you want to define how others may use, modify, and distribute this project.
