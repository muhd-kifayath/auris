# Agent for Unstructured Retrieval and Intelligent Synthesis  
## A Multi-Modal Document Digitization and RAG System for NEET Question Generation

---

### 🧠 Overview

This project is a Retrieval-Augmented Generation (RAG) system designed specifically for **NEET (National Eligibility cum Entrance Test)** preparation. It takes unstructured documents (text, scanned notes, PDFs, etc.), vectorizes and indexes them, and uses GPT-4o to generate exam-style MCQs. The agent uses intelligent synthesis via GPT models to ensure that the generated content remains **factually consistent**, **context-aware**, and **exam-relevant**.

---

### 🎯 Key Features

- 📄 **Multi-Modal Input**: Works with raw text, extracted PDF text, scanned notes (with OCR), and structured content.
- 🔍 **Contextual RAG**: Retrieves the most relevant chunks using sentence-level embeddings before generation.
- 🧬 **NEET-Specific MCQs**: Questions are crafted to align with the NEET syllabus and past-year patterns.
- 📌 **Summary-Based Querying**: Uses a GPT-generated summary of the topic to retrieve richer contextual documents.
- 🤖 **Dual Generation Modes**:
  - `Without RAG`: Generates questions from prior knowledge only.
  - `With RAG`: Uses vector-based context from digitized documents.
- 📂 **Output Format**: Questions are saved as structured JSON files under the `evaluation/` folder.

---

### ⚙️ Pipeline

1. **Topic Input**  
   Provide a NEET-relevant topic, such as `"Biological Classification"`.

2. **Summary Generation**  
   A concise summary of the topic is created using GPT-4o-mini to retrieve better document context.

3. **Vector Retrieval**  
   The summary is encoded with `sentence-transformers`, and matched documents are retrieved using cosine similarity from a pre-built vector store.

4. **Question Generation**  
   GPT-4o-mini generates NEET-style MCQs with explanations based on either:
   - No context (baseline)
   - Retrieved context (RAG-enhanced)

5. **Storage**  
   Questions are saved as:
   - `evaluation/topic_without_rag.json`
   - `evaluation/topic_with_rag.json`

---

### 🧪 Sample Output Format

```json
[
  {
    "question": "Which of the following kingdoms includes prokaryotic organisms?",
    "options": {
      "A": "Plantae",
      "B": "Animalia",
      "C": "Monera",
      "D": "Fungi"
    },
    "answer": "C",
    "explanation": "The kingdom Monera consists of prokaryotic organisms, including bacteria."
  }
]
```

---

### 📚 Technologies Used

- **OpenAI GPT-4o-mini** — LLM for generation and summarization  
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — Embedding model for vector search  
- **Chroma / FAISS** — Vector storage and retrieval  
- **Python** — Core implementation language  
- **JSON** — Question storage format for further evaluation
- **PyMuPDF** — Parse PDF Document
- **Docx** — Parse Word Document

---

### 🚀 Extensibility

- 🔍 Replaceable vector store backend (e.g., FAISS, ChromaDB)
- 📊 Extend to support image-based documents via OCR (e.g., Tesseract)
- 🧠 Can support evaluation metrics like BLEU, ROUGE, hallucination score, etc.
- 📘 Can be paired with a frontend for interactive quizzes

---

### ✅ Current Focus

- Ensuring factual consistency and syllabus alignment
- High-quality, diverse question generation for NEET topics
- Evaluating GPT-generated NEET questions using both manual and automated metrics (planned)

---

### 👨‍🔬 Ideal For

- EdTech platforms building NEET content pipelines  
- Researchers working on LLM + retrieval hybrid systems  
- Automating NEET-style question creation for scalable test generation  

---

### ✍️ Authors & Contributors

Built with ❤️ for AI-assisted education and exam prep.

---

