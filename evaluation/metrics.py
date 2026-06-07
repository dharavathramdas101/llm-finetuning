import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(prediction: str, reference: str) -> float:
    ref_tokens = [reference.split()]
    pred_tokens = prediction.split()
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    return round(score, 4)


def compute_all_metrics(prediction: str, reference: str) -> dict:
    rouge = compute_rouge(prediction, reference)
    bleu = compute_bleu(prediction, reference)
    return {"bleu": bleu, **rouge}
