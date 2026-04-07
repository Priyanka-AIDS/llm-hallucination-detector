# 🔍 RUC-Detect: Modular Hallucination Detection

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official repository for **"Detecting LLM Hallucinations via Fused Internal Probing and External Grounding"**.

## Quickstart

### 1. Environment Setup (environment.yml inside repository)
We provide a zero-conflict Conda environment:
```bash
conda env create -f environment.yml
conda activate ruc-detect
```
Alternatively, using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Download Checkpoints
Run the download script to fetch the Meta-Learner weights and the Llama-3 probe matrices:
```bash
python scripts/download_weights.py --target all
```
*Note: Checkpoints are also available on our HuggingFace Model Hub repository.*

### 3. Run the Gradio Demo
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

### 4. Deploy the API via Docker
To run the production-ready FastAPI service:
```bash
docker build -t ruc-detect-api .
docker run -p 8000:8000 --gpus all ruc-detect-api
```
Access the interactive OpenAPI Docs at: `http://localhost:8000/docs`.

---

## 🔬 Reproducing Main Results

To reproduce Table 1 from our ablation study on the HaluEval test split, use the included evaluation script. All hyperparameters are hardcoded to match the paper exactly.

```bash
python evaluate.py \
    --dataset halueval \
    --split test \
    --batch_size 16 \
    --use_fusion true
```

### Hyperparameters Documented
- **Probe Layer:** `layer_16` (middle section of Llama-3-8B)
- **MC-Dropout Samples:** `N=10`
- **FAISS Top-K:** `K=3`
- **Fusion Meta-Learner Threshold:** `0.52` (Optimized for F1)
