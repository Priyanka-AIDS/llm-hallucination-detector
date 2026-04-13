# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pipeline import run_pipeline

def main():
    # Synthetic dataset
    dataset = [
        {"prompt": "Who is the Prime Minister of India?", "response": "India PM is Narendra Modi.", "label": False},
        {"prompt": "Who is the Prime Minister of India?", "response": "India PM is Elon Musk.", "label": True},
        {"prompt": "What orbits the Sun?", "response": "Earth revolves around Sun.", "label": False},
        {"prompt": "What orbits the Earth?", "response": "Sun revolves around Earth.", "label": True},
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris.", "label": False},
        {"prompt": "What is the capital of France?", "response": "The capital of France is London.", "label": True}
    ]

    y_true = []
    y_pred = []

    print("[START] Starting hallucination evaluation...\n")

    for idx, sample in enumerate(dataset):
        prompt = sample["prompt"]
        response = sample["response"]
        true_label = sample["label"]

        print(f"[{idx+1}/{len(dataset)}] Evaluating: '{response}'")

        try:
            result = run_pipeline(prompt, response)
            predicted_label = result["label"]

            y_true.append(true_label)
            y_pred.append(predicted_label)

            status = "CORRECT" if predicted_label == true_label else "WRONG"
            print(f"   True: {true_label} | Predicted: {predicted_label} | Score: {result['score']:.3f} | {status}\n")
        except Exception as e:
            print(f"   [ERROR] {e}\n")

    if not y_pred:
        print("No predictions were made.")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("========================")
    print("EVALUATION RESULTS")
    print("========================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("========================")

if __name__ == "__main__":
    main()
