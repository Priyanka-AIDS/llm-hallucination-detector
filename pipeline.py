from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import spacy

from utils.nli import get_nli_score
from utils.selfcheck import selfcheck_nli
from utils.taxonomy import classify_pattern

model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

index = faiss.read_index("models/faiss.index")

with open("models/docs.pkl", "rb") as f:
    documents = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_pipeline(prompt, response, sampled_responses=None):

    response_embedding = model.encode([response]).astype("float32")
    response_embedding = response_embedding / np.linalg.norm(
        response_embedding, axis=1, keepdims=True
    )

    D, I = index.search(response_embedding, k=5)

    top_docs = I[0]

    retrieved_docs_text = [documents[idx] for idx in top_docs]
    doc_embeddings = model.encode(retrieved_docs_text).astype("float32")
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    similarities = [
        float(cosine_similarity(response_embedding[0], doc_embeddings[i]))
        for i in range(len(doc_embeddings))
    ]

    avg_sim = float(np.mean(similarities))
    uncertainty = float(np.var(similarities))

    # ================= Claim Extraction =================
    doc = nlp(response)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # ================= NLI (FactCC idea) =================
    nli_scores = []

    for sent in sentences:
        for idx in top_docs:
            label, score = get_nli_score(documents[idx], sent)

            if label == "ENTAILMENT":
                nli_scores.append(score)
            elif label == "CONTRADICTION":
                nli_scores.append(0)

    nli_score = float(np.mean(nli_scores)) if nli_scores else 0.0

    # ================= SelfCheckGPT =================
    selfcheck_score = 0.0

    if sampled_responses and len(sampled_responses) >= 3:
        _, sc_scores = selfcheck_nli(response, sampled_responses)
        selfcheck_score = float(np.mean(sc_scores))

    # ================= Final Score =================
    retrieval_score = (1 - avg_sim) + 0.2 * uncertainty

    final_score = min(
        1.0,
        0.4 * retrieval_score +
        0.3 * (1 - nli_score) +
        0.3 * selfcheck_score
    )

    label = final_score > 0.52

    # ================= Taxonomy =================
    pattern, explanation = classify_pattern(avg_sim, nli_score, final_score)

    # ================= Span Detection =================
    flagged_spans = []

    for sent in sentences:
        emb = model.encode([sent]).astype("float32")
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        D, I_span = index.search(emb, k=3)
        span_top_docs = I_span[0]

        span_retrieved_docs = [documents[idx] for idx in span_top_docs]
        span_doc_embeddings = model.encode(span_retrieved_docs).astype("float32")
        span_doc_embeddings = span_doc_embeddings / np.linalg.norm(span_doc_embeddings, axis=1, keepdims=True)

        sims = [
            float(cosine_similarity(emb[0], span_doc_embeddings[i]))
            for i in range(len(span_doc_embeddings))
        ]

        avg = float(np.mean(sims)) if sims else 0.0

        if avg < 0.5:
            start = response.find(sent)
            flagged_spans.append({
                "start": start,
                "end": start + len(sent),
                "text": sent,
                "confidence": float(1 - avg)
            })

    return {
        "score": float(final_score),
        "label": bool(label),
        "explanation": explanation,
        "pattern": pattern,
        "spans": flagged_spans,
        "components": {
            "retrieval_similarity": float(avg_sim),
            "nli_score": float(nli_score),
            "selfcheck_score": float(selfcheck_score)
        }
    }