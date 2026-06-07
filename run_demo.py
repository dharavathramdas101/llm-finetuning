"""
Local equivalent of Colab Cell 9.
Side-by-side comparison: base vs fine-tuned Ministral 3B.
Run after placing adapter files in outputs/model/
"""
import sys
sys.path.insert(0, ".")
from inference.run_model import load_base_model, load_finetuned_model, generate_response

QUESTIONS = [
    "What is your return policy?",
    "How do I track my order?",
    "I was charged twice for my order. What should I do?",
]

print("Loading base model...")
base_model, base_tok = load_base_model()
print("Loading fine-tuned model...")
ft_model, ft_tok = load_finetuned_model()

for q in QUESTIONS:
    print(f"\n{'='*65}")
    print(f"Q: {q}")
    print(f"{'='*65}")
    print("BASE:")
    print(generate_response(base_model, base_tok, q))
    print("\nFINE-TUNED:")
    print(generate_response(ft_model, ft_tok, q))
