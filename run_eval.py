"""
Local equivalent of Colab Cell 8.
Run after placing adapter files in outputs/model/
"""
import sys
sys.path.insert(0, ".")
from evaluation.evaluate import evaluate_both_models

evaluate_both_models()
