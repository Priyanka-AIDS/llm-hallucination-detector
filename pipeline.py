from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 🔥 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📚 STEP 1: KNOWLEDGE BASE
# =========================

documents = [
    "The Titanic sank in 1912 in the Atlantic Ocean.",
    "COVID-19 mRNA vaccines do not alter human DNA.",
    "The Earth revolves around the Sun.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The capital of France is Paris."
]

# =========================
# 📊 STEP 2: CREATE FAISS INDEX
# =========================

doc_embeddings = model.encode(documents)
dimension = doc_embeddings.shape[1]

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

    # 🔍 Retrieve top-2 relevant docs
    D, I = index.search(np.array(response_embedding), k=2)

    retrieved_docs = [documents[i] for i in I[0]]

    # 🧠 Compare similarity with retrieved docs
    similarities = []
    for doc in retrieved_docs:
        doc_emb = model.encode(doc)
        sim = cosine_similarity(response_embedding[0], doc_emb)
        similarities.append(sim)

    max_sim = max(similarities)

    # 🎯 Hallucination score
    score = float(1 - max_sim)

    explanation = f"Compared with knowledge base. Similarity: {max_sim:.2f}"

    return {
        "score": score,
        "label": score > 0.5,
        "explanation": explanation,
        "spans": [],
        "components": {
            "retrieval_similarity": float(max_sim)
        }
    }