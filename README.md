# AI Document Assistant (RAG)

A simple Streamlit RAG app that lets you ask questions about documents. It supports local PDF, DOCX, TXT, and Markdown uploads, plus public/shared Google Drive file or folder links.

## Features

- Extracts PDF text page by page; DOCX, TXT, and MD text with filename metadata.
- Splits text into overlapping chunks and preserves filename/page metadata.
- Creates Sentence Transformers embeddings using `all-MiniLM-L6-v2`.
- Uses FAISS for semantic retrieval and combines it with keyword matching.
- Sends retrieved context to Groq and instructs the model to answer only from that context.
- Displays retrieved source chunks after each answer.
- Keeps processed documents, chunks, and the FAISS index in Streamlit session state so a question does not re-embed the documents.
- Supports Google Drive links for files/folders shared in a way that permits downloading.

## Project files

```text
rag-ai-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Create a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

Install packages:

```powershell
pip install -r requirements.txt
```

## 2. Add your Groq API key

Create this file locally:

```text
.streamlit/secrets.toml
```

Put your key in it:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit `secrets.toml` or publish your API key. Add `.streamlit/secrets.toml` to `.gitignore` if you use Git.

For Streamlit Community Cloud, open your app's **Settings / Secrets** and add the same TOML entry:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

## 3. Run locally

```powershell
streamlit run app.py
```

Streamlit will show a local URL in the terminal.

## 4. Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Add `GROQ_API_KEY` in the app's Secrets settings.
5. Deploy.

Never upload your API key or `secrets.toml` to GitHub.

## How the RAG pipeline works

1. **Extract:** read text from each supported file and preserve its filename and page where available.
2. **Chunk:** split text into 900-character pieces with 150-character overlap.
3. **Embed:** turn each chunk into a vector using Sentence Transformers.
4. **Index:** add normalized vectors to a FAISS inner-product index (cosine similarity).
5. **Retrieve:** embed the question, run FAISS semantic search, and calculate keyword overlap.
6. **Generate:** send the top matching chunks and question to Groq. The prompt tells the model to rely only on retrieved context.
7. **Show sources:** display the filename, page if available, and retrieved chunk text.

## Notes and limitations

- The first run downloads the Sentence Transformers model and may take a few minutes.
- The embedding model runs locally and can use substantial RAM. A small deployment instance may need extra memory.
- PDF text extraction does not perform OCR. Scanned/image-only PDFs need OCR before their text can be searched.
- DOCX does not reliably expose page numbers because pagination depends on the word processor and layout. The app shows the filename and leaves page blank for DOCX, TXT, and MD.
- Google Drive access depends on the file/folder sharing permissions and gdown's supported link formats. Private files requiring a Google account login may not download. Use a publicly accessible or link-shared Drive resource.
- Documents and vectors are held in Streamlit session state for the active session. Restarting the app or starting a new session requires loading the documents again.
- Groq model availability can change. If `llama-3.3-70b-versatile` is unavailable for your account, change the model name in `answer_with_groq()` to a model currently enabled in your Groq console.
