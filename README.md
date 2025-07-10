# ML-Web-Interface

# 🤖 Mistral-7B Local LLM API (via Colab)

This project enables you to run a powerful **Mistral-7B-Instruct** model locally on **Google Colab (GPU)** and access it through a lightweight **FastAPI** server. It connects seamlessly with a **Streamlit frontend** for question-answering tasks — all without any OpenAI API key required.

---

## 🚀 Features

- Runs **Mistral-7B-Instruct** using `ctransformers` in Google Colab
- Exposes a `/generate` API using **FastAPI**
- Automatically uses GPU acceleration via Colab
- Optionally supports **Retrieval-Augmented Generation (RAG)** using local docs
- Can be accessed from a local Streamlit app via a public tunnel (e.g., LocalTunnel)

---

## 📦 Requirements (handled in Colab)

- `ctransformers`
- `fastapi`
- `uvicorn`
- `nest_asyncio`
- `localtunnel` (via `!npx localtunnel`)

---

## 🟩 Run on Google Colab

Click the badge below to launch:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zmengjie/ML-Web-Interface/blob/main/mistral_llm_api_colab.ipynb)

---

## 🛠️ How It Works

1. Loads `Mistral-7B-Instruct` using `ctransformers` with GPU
2. Launches a FastAPI app to serve `POST /generate` endpoint
3. Opens a public tunnel using LocalTunnel (or ngrok)
4. Your **Streamlit frontend** sends POST requests to the Colab URL to get responses

---

## 📄 Example Request

```bash
POST /generate
Content-Type: application/json

{
  "query": "What is Newton's Method in optimization?"
}
