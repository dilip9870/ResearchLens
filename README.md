# ResearchLens — RAG-Based Research Paper Assistant

A local desktop web app for asking questions about uploaded research paper PDFs. The app retrieves relevant passages with vector search and sends those passages to a locally running LLM, then shows the answer and retrieved page references.


## Stack
- **Python** — application and processing logic
- **Streamlit** — local web UI
- **PyMuPDF (fitz)** — PDF text extraction, with page numbers
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — semantic embeddings
- **FAISS** — similarity search over embedded text chunks
- **Ollama + Llama 3.2** — local response generation


## Requirements
- Windows 10/11
- Python 3.10 or newer (Python 3.11 recommended)
- Ollama installed and running
- Internet access for the first package/model downloads
- A text-selectable research PDF (scanned PDFs need OCR first)


## Setup and launch (Windows)
1. Install Python from the official Python website. During setup, enable **Add Python to PATH**.
2. Install Ollama and open it so its local service is running.
3. Open Command Prompt and run:
   ```bat
   ollama pull llama3.2
   ```
4. Extract this project folder if it came as a ZIP.
5. Double-click **START_RESEARCHLENS.bat**. It creates a virtual environment, installs Python dependencies, and starts the app.
6. Your browser opens (or Streamlit prints a local address, commonly `http://localhost:8501`). The app runs on your computer.

For later runs, double-click **RUN_RESEARCHLENS.bat**.


## How to use
1. Upload a research PDF.
2. Wait while ResearchLens extracts text, splits it into overlapping chunks, embeds the chunks, and builds a FAISS index.
3. Ask a question in the chat box.
4. ResearchLens searches the closest passages and asks the local Llama model to answer using the passages.
5. Expand **Retrieved passages** to review the supporting text and page numbers.


## Settings
The sidebar allows you to change:
- Ollama URL (default `http://localhost:11434`)
- Ollama model name (default `llama3.2`)
- Embedding model (default `sentence-transformers/all-MiniLM-L6-v2`)
- Retrieval passage count (top-k)
- Chunk size and overlap
- Answer creativity (temperature)


## Troubleshooting
- **Ollama connection error:** Open Ollama, then try `ollama list` in Command Prompt.
- **Model not found:** Run `ollama pull llama3.2`; ensure the sidebar model name matches `ollama list`.
- **First run takes time:** Python packages and the embedding model are downloaded once; allow a few minutes.
- **PDF has no selectable text:** This version does not perform OCR. OCR the PDF before uploading.
- **Slow answers:** Smaller Ollama models may respond faster; use a smaller model supported by your Ollama installation and enter its exact model name in the sidebar.
- **Port in use:** Streamlit may choose a different local port or show a message. Follow the address it prints.

## Project structure
```
ResearchLens/
├── app.py
├── requirements.txt
├── README.md
├── START_RESEARCHLENS.bat
└── RUN_RESEARCHLENS.bat
```

## Notes
- This is a local-first project. Your PDF and questions are processed by this app and the local Ollama service, rather than sent to a hosted LLM API by the app.
- Answers can still be incorrect. Always verify important statements against the cited passages in the paper.
- If the PDF has no text layer, OCR it first with a separate OCR tool.
