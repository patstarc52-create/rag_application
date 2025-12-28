import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
import gdown

from groq import Groq
from sentence_transformers import SentenceTransformer

# ===============================
# CONFIG
# ===============================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in Hugging Face Secrets")

FILE_ID = "1idPBjLYvfyTF7gsoBmLohcfLRrFFNaAC"
LOCAL_CSV = "knowledge.csv"

TOP_K = 5
MAX_DISTANCE = 2.5

# ===============================
# LOAD MODELS
# ===============================
client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# DOWNLOAD CSV (ONCE)
# ===============================
if not os.path.exists(LOCAL_CSV):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        LOCAL_CSV,
        quiet=False
    )

# ===============================
# LOAD KNOWLEDGE BASE
# ===============================
df = pd.read_csv(LOCAL_CSV)
documents = df.astype(str).agg(" | ".join, axis=1).tolist()

# ===============================
# BUILD FAISS INDEX
# ===============================
embeddings = embed_model.encode(documents, convert_to_numpy=True).astype("float32")

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)

print(f"✅ Knowledge Base Loaded: {len(documents)} rows")

# ===============================
# RAG FUNCTION
# ===============================
def ask_question(question):
    if not question.strip():
        return "⚠️ Please ask a valid question."

    query_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = faiss_index.search(query_emb, TOP_K)

    if distances[0][0] > MAX_DISTANCE:
        return """
<div class='answer-box'>
<h2>Answer</h2>
<p><b>I don’t know.</b><br>This question is outside my knowledge base.</p>
</div>
"""

    context = "\n".join(documents[i] for i in indices[0])

    prompt = f"""
You are a strict knowledge-based assistant.
Answer ONLY using the context.
If the answer is not present, say "I don’t know".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    return f"""
<div class='answer-box'>
<h2>Answer</h2>
<p>{answer}</p>
</div>
"""

# ===============================
# UI
# ===============================
css = """
body { background-color: #0f172a; color: #f9fafb; }
.answer-box {
  background-color: #020617;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 18px;
}
.answer-box h2 { color: #3b82f6; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🧠 Knowledge Base Assistant")
    gr.Markdown("Ask questions strictly from the internal knowledge base.")

    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask something from the knowledge base…",
        lines=2
    )

    answer = gr.Markdown(
        "<div class='answer-box'><h2>Answer</h2><p>Your answer will appear here.</p></div>"
    )

    question.submit(ask_question, question, answer)

demo.launch()
