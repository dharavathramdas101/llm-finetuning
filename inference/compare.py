import sys
import textwrap

sys.path.insert(0, ".")
from inference.run_model import load_base_model, load_finetuned_model, generate_response


def wrap(text: str, width: int = 60) -> str:
    return "\n".join(textwrap.wrap(text, width))


def main():
    if len(sys.argv) < 2:
        question = input("Enter question: ").strip()
    else:
        question = " ".join(sys.argv[1:])

    print("\nLoading models (may take ~1 min)...")
    base_model, base_tok = load_base_model()
    ft_model, ft_tok = load_finetuned_model()

    base_resp = generate_response(base_model, base_tok, question)
    ft_resp = generate_response(ft_model, ft_tok, question)

    sep = "-" * 62
    print(f"\n{'Question':>10}: {question}")
    print(f"\n{sep}")
    print("BASE PHI-2")
    print(sep)
    print(wrap(base_resp))
    print(f"\n{sep}")
    print("FINE-TUNED (QLoRA)")
    print(sep)
    print(wrap(ft_resp))
    print()


if __name__ == "__main__":
    main()
