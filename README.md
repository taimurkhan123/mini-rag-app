# 📚 Mini RAG — Ask Questions from Your PDF

An interactive **Retrieval-Augmented Generation (RAG)** application that allows users to upload custom PDF documents and ask questions grounded strictly in the document context.

Built with **Python**, **Streamlit**, **Sentence-Transformers**, and **Groq AI**.

---

## 👨‍💻 Author

* **Developer:** Taimur Khan
* **Project:** Mini RAG Application

---

## ⭐ Key Features

* **📄 Instant PDF Ingestion:** Extract text from any digital PDF document in seconds.
* **⚡ Semantic Search:** Uses local vector embeddings (`all-MiniLM-L6-v2`) to perform cosine similarity matching on document chunks.
* **🤖 AI-Powered Answers:** Integrates with Groq's high-speed inference engine (`llama-3.3-70b-versatile`) to deliver clear, precise answers.
* **🛡️ Hallucination Guardrails:** Strictly restricts the model to the provided document context to prevent fake or off-topic responses.
* **🎛️ Interactive Controls:** Features customizable sliders for chunk size, overlap, and retrieval top-$k$ parameters.
* **🔍 Transparent Citations:** Expandable view showing exact retrieved source chunks alongside similarity scores for every answer.

---

## 💡 How It Works

1. **Upload & Process:** Upload a PDF; the app splits text into configurable chunks and generates vector embeddings locally.
2. **Search:** Enter a question. The app converts your prompt into an embedding vector and finds the top matching document chunks.
3. **Generate:** The retrieved chunks are sent to the LLM via Groq API to construct an accurate, context-bound answer.

---

## 📬 Connect & Support

If you have any feedback, questions, or ideas for improvement, feel free to reach out or open an issue on GitHub!
## Live URL  = https://my-small-rag-app.streamlit.app/
