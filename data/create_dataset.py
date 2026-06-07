from datasets import load_dataset
import json
import random
import os

def create_dataset():
    print("Downloading bitext customer support dataset...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        trust_remote_code=True
    )

    data = dataset["train"].shuffle(seed=42)
    data = data.select(range(1000))

    formatted = []
    for item in data:
        formatted.append({
            "instruction": item["instruction"],
            "input": "",
            "output": item["response"]
        })

    random.seed(42)
    random.shuffle(formatted)

    train = formatted[:800]
    val   = formatted[800:900]
    test  = formatted[900:]

    os.makedirs("data/processed", exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = f"data/processed/{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"{split_name.capitalize()}: {len(split_data)} pairs → {path}")

    print("\nDataset ready.")


if __name__ == "__main__":
    create_dataset()
