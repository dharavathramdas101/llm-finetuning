import sys
import json
import os

sys.path.insert(0, ".")
from inference.run_model import load_base_model, load_finetuned_model, generate_response
from evaluation.metrics import compute_all_metrics
import pandas as pd


def evaluate_both_models(n_samples: int = 50):
    print("Loading base model...")
    base_model, base_tok = load_base_model()

    print("Loading fine-tuned model...")
    ft_model, ft_tok = load_finetuned_model()

    test_data = []
    with open("data/processed/test.jsonl", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))

    samples = test_data[:n_samples]
    results = []

    for i, item in enumerate(samples):
        question = item["instruction"]
        reference = item["output"]

        base_resp = generate_response(base_model, base_tok, question)
        ft_resp = generate_response(ft_model, ft_tok, question)

        base_m = compute_all_metrics(base_resp, reference)
        ft_m = compute_all_metrics(ft_resp, reference)

        results.append({
            "question": question,
            "reference": reference,
            "base_response": base_resp,
            "ft_response": ft_resp,
            "base_bleu":   base_m["bleu"],
            "ft_bleu":     ft_m["bleu"],
            "base_rouge1": base_m["rouge1"],
            "ft_rouge1":   ft_m["rouge1"],
            "base_rouge2": base_m["rouge2"],
            "ft_rouge2":   ft_m["rouge2"],
            "base_rougeL": base_m["rougeL"],
            "ft_rougeL":   ft_m["rougeL"],
        })

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{n_samples}")

    df = pd.DataFrame(results)

    print("\n=== RESULTS ===")
    print(f"{'Metric':<14} {'Base':>8} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 46)
    for col, label in [("bleu","BLEU"), ("rouge1","ROUGE-1"), ("rouge2","ROUGE-2"), ("rougeL","ROUGE-L")]:
        b = df[f"base_{col}"].mean()
        f = df[f"ft_{col}"].mean()
        print(f"{label:<14} {b:>8.4f} {f:>12.4f} {f-b:>+8.4f}")

    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/evaluation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")

    return df


if __name__ == "__main__":
    evaluate_both_models()
