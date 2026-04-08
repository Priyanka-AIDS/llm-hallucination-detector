from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset

# =========================
# 🔥 Load embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📚 STEP 1: KNOWLEDGE BASE
# =========================

# Use stable dataset (AG News)
dataset = load_dataset("ag_news", split="train[:2000]")

# Extract text
documents = [x["text"] for x in dataset]

# =========================
# 📊 STEP 2: CREATE FAISS INDEX
# =========================

# Encode documents
doc_embeddings = model.encode(documents)

# Normalize embeddings (for cosine similarity)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

dimension = doc_embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# =========================
# 🧠 HELPER FUNCTION
# =========================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# 🚀 MAIN PIPELINE
# =========================

def run_pipeline(prompt, response):
    # Encode response
    response_embedding = model.encode([response])

    # Normalize response embedding
    response_embedding = response_embedding / np.linalg.norm(
        response_embedding, axis=1, keepdims=True
    )

    # 🔍 Retrieve top-5 similar documents
    D, I = index.search(np.array(response_embedding), k=5)

    retrieved_docs = [documents[i] for i in I[0]]

    # 🧠 Compute similarity
    similarities = []
    for idx in I[0]:
        doc_emb = doc_embeddings[idx]
        sim = cosine_similarity(response_embedding[0], doc_emb)
        similarities.append(sim)

    # Safe average
    avg_sim = sum(similarities) / len(similarities) if similarities else 0

    # 🎯 Hallucination score
    score = float(1 - avg_sim)

    explanation = f"Avg similarity with knowledge base: {avg_sim:.2f}"

    return {
        "score": score,
        "label": score > 0.5,
        "explanation": explanation,
        "spans": [],  # will improve later
        "components": {
            "retrieval_similarity": float(avg_sim)
        }
    }