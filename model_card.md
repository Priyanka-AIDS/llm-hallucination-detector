---
language: en
tags:
- hallucination-detection
- factual-consistency
- verification
- probe
license: apache-2.0
datasets:
- halueval
- truthful_qa
- factscore
metrics:
- f1
- roc_auc
---

# Model Card for RUC-Detect (Fusion Meta-Learner)

## Model Details
- **Developer:** NLP Research Team
- **Model Type:** Multi-modal Fusion Classifier + Transformer Probe
- **Language(s):** English
- **License:** Apache 2.0
- **Model Description:** RUC-Detect fuses internal hidden-state probing with external RAG entailment to predict the probability that an LLM's generated response contains an intrinsic, extrinsic, or semantic drift hallucination.

## Intended Use
- **Primary Use Cases:** Real-time monitoring of LLM outputs in production, automated verification of generative summarization, and human-in-the-loop review assistance.
- **Out-of-Scope Use Cases:** Not intended for zero-shot factuality checking on unsupported languages, nor evaluating non-generative tasks like sentiment analysis. It should not be the sole decision-maker in life-critical medical scenarios.

## Training Data
Trained on a highly curated 75K synthetically augmented dataset composed of rewritten samples from HaluEval, SQuAD 2.0, and XSum. The dataset ensures a perfectly balanced, stratified taxonomy of hallucinations.

## Evaluation Results
- **TruthfulQA (MC2):** 79.5%
- **FactScore:** 84.4 F1
- **HaluEval (AUC):** 81.2%
- **Inference Latency:** ~850ms on single RTX 3090.

## Bias, Risks, and Limitations
The model heavily relies on the quality of the external retrieval database. If the FAISS index contains misinformation, the RAG entailment component may confidently flag factual statements as hallucinations (False Positives). Furthermore, the geometric probe was trained exclusively on the Llama-3 architecture; utilizing hidden states from drastically diverse architectures (e.g., MoE models) requires recalibrating the meta-learner.

## Citation
```bibtex
@misc{nlp_research_2024_rucdetect,
  title={Detecting LLM Hallucinations via Fused Internal Probing and External Grounding},
  author={NLP Research Team},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/models/nlp_research/ruc-detect}}
}
```
