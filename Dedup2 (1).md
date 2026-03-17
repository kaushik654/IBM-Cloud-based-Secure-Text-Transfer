Two errors visible:

1. `BLEUMetric.a_measure() got unexpected keyword argument '_show_indicator'` — your `a_measure` signature doesn't accept `**kwargs`
2. `AttributeError: 'Scorer' object has no attribute 'bleu_score'` — you're still using the old `Scorer()`-based code in that cell

---

## Fix: Replace the entire metrics cell with this clean version

This uses **only `nltk` and `rouge_score`** — no `Scorer()` at all:

```python
import math
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer as rs
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Perplexity ────────────────────────────────────────────────────────────────
class PerplexityMetric(BaseMetric):
    def __init__(self, threshold: float = 100.0):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        inputs = tokenizer(test_case.actual_output, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss
        self.score = math.exp(loss.item())
        self.success = self.score < self.threshold
        self.reason = f"Perplexity={self.score:.2f}, threshold={self.threshold}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:  # ← **kwargs fixes error 1
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Perplexity"


# ── BLEU ──────────────────────────────────────────────────────────────────────
class BLEUMetric(BaseMetric):
    def __init__(self, threshold: float = 0.1, bleu_type: str = "bleu4"):
        super().__init__()
        self.threshold = threshold
        self.bleu_type = bleu_type           # ← string, not variable
        self.score = 0.0
        self.success = False
        self.reason = ""
        weights_map = {
            "bleu1": (1.0, 0.0, 0.0, 0.0),
            "bleu2": (0.5, 0.5, 0.0, 0.0),
            "bleu3": (0.33, 0.33, 0.33, 0.0),
            "bleu4": (0.25, 0.25, 0.25, 0.25),
        }
        self.weights = weights_map.get(bleu_type, (0.25, 0.25, 0.25, 0.25))
        self.smoothing = SmoothingFunction().method1

    def measure(self, test_case: LLMTestCase) -> float:
        hyp = word_tokenize(test_case.actual_output.lower())
        ref = word_tokenize(test_case.expected_output.lower())
        self.score = sentence_bleu(
            [ref], hyp,
            weights=self.weights,
            smoothing_function=self.smoothing
        )
        self.success = self.score >= self.threshold
        self.reason = f"BLEU={self.score:.4f}, threshold={self.threshold}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:  # ← **kwargs
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return f"BLEU ({self.bleu_type})"


# ── ROUGE ─────────────────────────────────────────────────────────────────────
class ROUGEMetric(BaseMetric):
    def __init__(self, threshold: float = 0.3, score_type: str = "rouge1"):
        super().__init__()
        self.threshold = threshold
        self.score_type = score_type
        self.score = 0.0
        self.success = False
        self.reason = ""
        self._scorer = rs.RougeScorer([score_type], use_stemmer=True)

    def measure(self, test_case: LLMTestCase) -> float:
        result = self._scorer.score(
            test_case.expected_output,
            test_case.actual_output
        )
        self.score = result[self.score_type].fmeasure
        self.success = self.score >= self.threshold
        self.reason = f"ROUGE={self.score:.4f}, threshold={self.threshold}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:  # ← **kwargs
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return f"ROUGE ({self.score_type})"
```

---

## Root causes summary

| Error | Cause | Fix |
|---|---|---|
| `unexpected keyword argument '_show_indicator'` | `a_measure` signature too strict | Add `**kwargs` to all `a_measure` methods |
| `Scorer has no attribute 'bleu_score'` | Old cell still using `Scorer()` | Remove `Scorer`, use `nltk` directly |
| Previous `bleu1 not defined` | Missing quotes | `self.bleu_type = bleu_type` |

The `**kwargs` fix is the critical one — DeepEval passes internal kwargs like `_show_indicator` and `_in_component` to `a_measure` which your signatures were rejecting.
