import os
import pickle
import numpy as np
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# -------------------------------
# STEP 0: Setup
# -------------------------------
print("🚀 Starting FAISS index build...")

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
print("📥 Loading dataset...")

dataset = load_dataset("ag_news", split="train[:2000]")

# Extract text
documents = [x["text"] for x in dataset]

print(f"✅ Loaded {len(documents)} documents")

# -------------------------------
# STEP 2: Load Embedding Model
# -------------------------------
print("🧠 Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# STEP 3: Generate Embeddings
# -------------------------------
print("⚡ Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

# Convert to numpy float32 (IMPORTANT for FAISS)
embeddings = np.array(embeddings).astype("float32")

print(f"✅ Embeddings shape: {embeddings.shape}")

# -------------------------------
# STEP 4: Build FAISS Index
# -------------------------------
print("📦 Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} vectors")

# -------------------------------
# STEP 5: Save Index
# -------------------------------
index_path = "models/faiss.index"
faiss.write_index(index, index_path)

print(f"💾 Saved FAISS index to {index_path}")

# -------------------------------
# STEP 6: Save Documents
# -------------------------------
docs_path = "models/docs.pkl"

with open(docs_path, "wb") as f:
    pickle.dump(documents, f)

print(f"💾 Saved documents to {docs_path}")

# -------------------------------
# DONE
# -------------------------------
print("🎉 SUCCESS: Index building completed!")