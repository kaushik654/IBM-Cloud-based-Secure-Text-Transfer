import math
import torch
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

nltk.download("punkt", quiet=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Qwen 0.6B

# ─── Load model once ──────────────────────────────────────────────────────────
print("Loading Qwen 0.6B model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()


# ─── 1. Perplexity Metric ─────────────────────────────────────────────────────
class PerplexityMetric(BaseMetric):
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        text = test_case.actual_output
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss  # cross-entropy = avg negative log-likelihood
        self.score = math.exp(loss.item())
        self.success = self.score < self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Perplexity"


# ─── 2. BLEU Metric ───────────────────────────────────────────────────────────
class BLEUMetric(BaseMetric):
    """
    Uses DeepEval's built-in Scorer.
    bleu_type: 'bleu1' | 'bleu2' | 'bleu3' | 'bleu4'
    """
    def __init__(self, threshold: float = 0.3, bleu_type: str = "bleu4"):
        self.threshold = threshold
        self.bleu_type = bleu_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase) -> float:
        self.score = self.scorer.bleu_score(
            prediction=test_case.actual_output,
            references=test_case.expected_output,  # str or list[str]
            bleu_type=self.bleu_type,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return f"BLEU ({self.bleu_type})"


# ─── 3. ROUGE Metric ──────────────────────────────────────────────────────────
class ROUGEMetric(BaseMetric):
    """
    score_type: 'rouge1' | 'rouge2' | 'rougeL'
    """
    def __init__(self, threshold: float = 0.4, score_type: str = "rouge1"):
        self.threshold = threshold
        self.score_type = score_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase) -> float:
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type=self.score_type,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return f"ROUGE ({self.score_type})"


# ─── Generate output from Qwen 0.6B ──────────────────────────────────────────
def generate_output(prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # strip the prompt tokens from output
    generated = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ─── Test Cases ───────────────────────────────────────────────────────────────
samples = [
    {
        "input": "What is the capital of France?",
        "expected": "The capital of France is Paris.",
    },
    {
        "input": "Explain what photosynthesis is in one sentence.",
        "expected": "Photosynthesis is the process by which plants use sunlight to convert CO2 and water into glucose and oxygen.",
    },
    {
        "input": "What is 2 + 2?",
        "expected": "2 + 2 equals 4.",
    },
]

test_cases = []
for s in samples:
    actual = generate_output(s["input"])
    print(f"\nInput   : {s['input']}")
    print(f"Expected: {s['expected']}")
    print(f"Actual  : {actual}")
    test_cases.append(
        LLMTestCase(
            input=s["input"],
            actual_output=actual,
            expected_output=s["expected"],
        )
    )

# ─── Run Evaluation ───────────────────────────────────────────────────────────
metrics = [
    PerplexityMetric(threshold=100.0),
    BLEUMetric(threshold=0.1, bleu_type="bleu4"),
    ROUGEMetric(threshold=0.3, score_type="rouge1"),
    ROUGEMetric(threshold=0.2, score_type="rouge2"),
    ROUGEMetric(threshold=0.3, score_type="rougeL"),
]

results = evaluate(test_cases, metrics)

# ─── Print Summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
for tc, metric in [(tc, m) for tc in test_cases for m in metrics]:
    metric.measure(tc)
    status = "✅ PASS" if metric.is_successful() else "❌ FAIL"
    print(f"[{metric.__name__}] Score: {metric.score:.4f}  {status}")
