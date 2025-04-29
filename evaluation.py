"""evaluation.py
A semantic evaluation module for grading handwritten (OCR-extracted) answers
against model answers. It supports three major question types:

1. theory   – Natural-language responses (conceptual / descriptive).
2. code     – Programming answers (source code snippets).
3. numerical – Pure numerical results (single numbers or short expressions).

The module leverages Sentence-Transformers for semantic similarity on text and
CodeBERT embeddings for code snippets, providing a cosine-similarity score in
the range [0, 1]. Numerical answers receive a relative-error-based score.

Dependencies
------------
pip install sentence-transformers torch transformers

Example
-------
>>> from evaluation import SemanticEvaluator
>>> ev = SemanticEvaluator()
>>> score = ev.compute_similarity("Photosynthesis occurs in chloroplasts", 
...                              "It happens inside the chloroplast", 
...                              question_type="theory")
>>> print(round(score, 3))
0.83
"""

from __future__ import annotations

import os
import ssl
import warnings
from typing import List, Sequence

import numpy as np

# Set environment variables to disable SSL verification for Hugging Face
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create a custom SSL context that doesn't verify certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Now import the sentence transformers
from sentence_transformers import SentenceTransformer, util


class SemanticEvaluator:
     """Compute semantic similarity scores between student answers and keys.

     Parameters
     ----------
     theory_model_name : str, optional
         Hugging Face model for natural-language answers. Defaults to an
         efficient MiniLM model.
     code_model_name : str, optional
         Model for source-code answers. Defaults to CodeBERT.
     device : str | None, optional
         Force model to specific device (e.g. "cuda", "cpu"). None = auto.
     """

     def __init__(
         self,
         theory_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
         code_model_name: str = "microsoft/codebert-base",
         device: str | None = None,
     ) -> None:
         # Try to load models with SSL verification disabled
         try:
             self.theory_model = SentenceTransformer(theory_model_name, device=device)
             # Code model loaded via SentenceTransformer wrapper for consistency
             self.code_model = SentenceTransformer(code_model_name, device=device)
         except Exception as e:
             print(f"Error loading models: {e}")
             print("Falling back to simple text comparison...")
             self.theory_model = None
             self.code_model = None

     # ------------------------------------------------------------------
     # Public API
     # ------------------------------------------------------------------
     def compute_similarity(
         self,
         student_answer: str,
         model_answer: str,
         *,
         question_type: str = "theory",
     ) -> float:
         """Return a similarity score in [0, 1] for a single Q-A pair."""
         question_type = question_type.lower()
         if question_type == "numerical":
             return self._numeric_similarity(student_answer, model_answer)
         if question_type == "code":
             return self._text_similarity(student_answer, model_answer, code=True)
         # default: theory
         return self._text_similarity(student_answer, model_answer, code=False)

     def batch_compute_similarity(
         self,
         student_answers: Sequence[str],
         model_answers: Sequence[str],
         question_types: Sequence[str] | None = None,
     ) -> List[float]:
         """Vectorised similarity for multiple answers. Lists must align 1-to-1."""
         if len(student_answers) != len(model_answers):
             raise ValueError("student_answers and model_answers must be same length")
         if question_types is None:
             question_types = ["theory"] * len(student_answers)
         if len(question_types) != len(student_answers):
             raise ValueError("question_types length mismatch")

         scores: List[float] = []
         for stu, key, qtype in zip(student_answers, model_answers, question_types):
             scores.append(self.compute_similarity(stu, key, question_type=qtype))
         return scores

     # ------------------------------------------------------------------
     # Internal helpers
     # ------------------------------------------------------------------
     def _text_similarity(self, s1: str, s2: str, *, code: bool = False) -> float:
         # If models failed to load, fall back to simple text comparison
         if self.theory_model is None or self.code_model is None:
             # Simple fallback using word overlap
             s1_words = set(s1.lower().split())
             s2_words = set(s2.lower().split())
             if not s1_words or not s2_words:
                 return 0.0
             # Jaccard similarity
             intersection = len(s1_words.intersection(s2_words))
             union = len(s1_words.union(s2_words))
             return intersection / union if union > 0 else 0.0
         
         # Use transformer models if available
         model = self.code_model if code else self.theory_model
         try:
             # encode returns torch.Tensor; convert_to_tensor ensures same device
             emb1 = model.encode(s1, convert_to_tensor=True)
             emb2 = model.encode(s2, convert_to_tensor=True)
             score = util.cos_sim(emb1, emb2).item()
             # cos_sim ∈ [-1,1]; scale to [0,1]
             return (score + 1) / 2
         except Exception as e:
             print(f"Error computing similarity: {e}")
             print("Falling back to simple text comparison...")
             # Simple fallback using word overlap
             s1_words = set(s1.lower().split())
             s2_words = set(s2.lower().split())
             if not s1_words or not s2_words:
                 return 0.0
             # Jaccard similarity
             intersection = len(s1_words.intersection(s2_words))
             union = len(s1_words.union(s2_words))
             return intersection / union if union > 0 else 0.0

     def _numeric_similarity(self, s_student: str, s_model: str) -> float:
         """Score numerical answers based on relative error (0 = worst, 1 = exact)."""
         try:
             student_val = float(eval(s_student.strip()))  # eval for simple exprs e.g., "3/4"
             model_val = float(eval(s_model.strip()))
         except Exception:
             # Fallback to language similarity if parsing fails
             return self._text_similarity(s_student, s_model)

         if model_val == 0:
             return 1.0 if student_val == 0 else 0.0
         rel_err = abs(student_val - model_val) / abs(model_val)
         return float(max(0.0, 1.0 - rel_err))


# ----------------------------------------------------------------------
# CLI utility (optional): python -m evaluation "student" "model" theory
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Semantic grading helper")
    parser.add_argument("student", help="Student answer text or path to file")
    parser.add_argument("model", help="Model answer text or path to file")
    parser.add_argument("type", choices=["theory", "code", "numerical"], nargs="?", default="theory")
    args = parser.parse_args()

    def _read_content(x: str) -> str:
        try:
            # Load as file if exists, else treat as direct string
            with open(x, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return x

    stu_txt = _read_content(args.student)
    key_txt = _read_content(args.model)

    evaluator = SemanticEvaluator()
    sim = evaluator.compute_similarity(stu_txt, key_txt, question_type=args.type)
    print(json.dumps({"similarity": sim}))
