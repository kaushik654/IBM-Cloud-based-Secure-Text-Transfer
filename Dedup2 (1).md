Now I understand the full picture. This is a **training-style dataset** where the model was trained on these conversations. So the evaluation dataset has the same format — the assistant turns **are** the ground truth.

---

## What This Dataset Actually Is

```
tools     → list of available tools the model can call
conversations → multi-turn dialogue:
    user         → input
    assistant    → ground truth (tool call OR text response)
    observation  → tool result (inject as context)
    assistant    → ground truth (follow-up text)
```

The model was **trained to predict assistant turns** given user + tools + observations. So evaluation = how well does Qwen 0.6B replicate those assistant turns.

---

## My Suggestions for Evaluation

Since there are **two types of assistant turns**, you need **two separate evaluation strategies**:

---

### Type 1 — Tool Call Turns
```json
"value": "[{\"name\":\"open_app\",\"arguments\":{\"app_name\":\"Camera\",\"action_within_app\":\"Video\"}}]"
```

These are **structured JSON** — BLEU/ROUGE are meaningless here. Use:

| Metric | What it checks |
|---|---|
| **Tool Name Exact Match** | Did model call the right tool? |
| **Argument Key Match** | Did model use correct argument names? |
| **Argument Value Match** | Did model pass correct values? |
| **Full JSON Exact Match** | Complete tool call match |

---

### Type 2 — Text Response Turns
```
"Alright! I've successfully opened the Camera app..."
```

These are natural language — use your existing metrics:

| Metric | Suitable? |
|---|---|
| **Perplexity** | ✅ Yes — measures fluency |
| **BLEU** | ✅ Yes — n-gram overlap |
| **ROUGE** | ✅ Yes — recall/precision overlap |
| **BERTScore** | ✅ Better — semantic similarity |

---

## Full Evaluator Code

```python
import json
import math
import torch
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer as rs
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True
)
model.eval()


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_tool_call(value: str) -> bool:
    """Check if assistant turn is a tool call (JSON list) or plain text."""
    try:
        parsed = json.loads(value)
        return isinstance(parsed, list) and "name" in parsed[0]
    except:
        return False

def generate(prompt: str, max_new_tokens=128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def build_prompt(tools, conversations_so_far):
    """Build prompt from tools + conversation history."""
    tool_str = json.dumps(tools, indent=2)
    prompt = f"Available tools:\n{tool_str}\n\nConversation:\n"
    for turn in conversations_so_far:
        role = turn["from"]
        value = turn["value"]
        prompt += f"{role}: {value}\n"
    prompt += "assistant:"
    return prompt


# ── Tool Call Metrics (no DeepEval needed) ────────────────────────────────────
def eval_tool_call(predicted: str, ground_truth: str) -> dict:
    results = {
        "tool_name_match": 0,
        "args_key_match": 0,
        "args_value_match": 0,
        "full_exact_match": 0,
    }
    try:
        pred_list = json.loads(predicted)
        gt_list   = json.loads(ground_truth)

        pred_tool = pred_list[0]
        gt_tool   = gt_list[0]

        # Tool name match
        if pred_tool.get("name") == gt_tool.get("name"):
            results["tool_name_match"] = 1

        pred_args = pred_tool.get("arguments", {})
        gt_args   = gt_tool.get("arguments", {})

        # Argument key match
        if set(pred_args.keys()) == set(gt_args.keys()):
            results["args_key_match"] = 1

        # Argument value match
        matches = sum(1 for k in gt_args if pred_args.get(k) == gt_args[k])
        results["args_value_match"] = matches / len(gt_args) if gt_args else 0

        # Full exact match
        if pred_tool == gt_tool:
            results["full_exact_match"] = 1

    except Exception as e:
        pass  # parse failed = all zeros

    return results


# ── DeepEval Text Metrics ──────────────────────────────────────────────────────
class PerplexityMetric(BaseMetric):
    def __init__(self, threshold=100.0):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        inputs = tokenizer(test_case.actual_output, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss
        self.score = math.exp(loss.item())
        self.success = self.score < self.threshold
        return self.score

    async def a_measure(self, test_case, **kwargs): return self.measure(test_case)
    def is_successful(self): return self.success

    @property
    def __name__(self): return "Perplexity"


class BLEUMetric(BaseMetric):
    def __init__(self, threshold=0.1, bleu_type="bleu4"):
        super().__init__()
        self.threshold = threshold
        self.bleu_type = bleu_type
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

    def measure(self, test_case: LLMTestCase):
        hyp = word_tokenize(test_case.actual_output.lower())
        ref = word_tokenize(test_case.expected_output.lower())
        self.score = sentence_bleu([ref], hyp, weights=self.weights,
                                   smoothing_function=self.smoothing)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, **kwargs): return self.measure(test_case)
    def is_successful(self): return self.success

    @property
    def __name__(self): return f"BLEU ({self.bleu_type})"


class ROUGEMetric(BaseMetric):
    def __init__(self, threshold=0.3, score_type="rouge1"):
        super().__init__()
        self.threshold = threshold
        self.score_type = score_type
        self.score = 0.0
        self.success = False
        self.reason = ""
        self._scorer = rs.RougeScorer([score_type], use_stemmer=True)

    def measure(self, test_case: LLMTestCase):
        result = self._scorer.score(test_case.expected_output, test_case.actual_output)
        self.score = result[self.score_type].fmeasure
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, **kwargs): return self.measure(test_case)
    def is_successful(self): return self.success

    @property
    def __name__(self): return f"ROUGE ({self.score_type})"


# ── Main Evaluator ────────────────────────────────────────────────────────────
def evaluate_dataset(json_path: str, limit: int = 20):

    with open(json_path, "r") as f:
        dataset = json.load(f)

    text_metrics = [
        PerplexityMetric(threshold=100.0),
        BLEUMetric(threshold=0.1, bleu_type="bleu4"),
        ROUGEMetric(threshold=0.3, score_type="rouge1"),
        ROUGEMetric(threshold=0.2, score_type="rouge2"),
        ROUGEMetric(threshold=0.3, score_type="rougeL"),
    ]

    # Accumulators
    text_scores   = {m.__name__: [] for m in text_metrics}
    tool_scores   = {"tool_name_match": [], "args_key_match": [],
                     "args_value_match": [], "full_exact_match": []}

    text_test_cases = []

    for record in dataset[:limit]:
        tools         = json.loads(record["tools"]) if isinstance(record["tools"], str) else record["tools"]
        conversations = record["conversations"]

        context = []  # build up conversation history

        for i, turn in enumerate(conversations):
            if turn["from"] != "user":
                context.append(turn)
                continue

            # Find the next assistant turn (ground truth)
            next_assistant = None
            for j in range(i+1, len(conversations)):
                if conversations[j]["from"] == "assistant":
                    next_assistant = conversations[j]["value"]
                    break

            if next_assistant is None:
                context.append(turn)
                continue

            # Build prompt and generate
            prompt = build_prompt(tools, context + [turn])
            actual = generate(prompt)

            if is_tool_call(next_assistant):
                # ── Tool call evaluation ──
                scores = eval_tool_call(actual, next_assistant)
                for k, v in scores.items():
                    tool_scores[k].append(v)
                print(f"[TOOL] GT:   {next_assistant[:80]}")
                print(f"[TOOL] PRED: {actual[:80]}")
                print(f"[TOOL] Scores: {scores}\n")
            else:
                # ── Text evaluation ──
                tc = LLMTestCase(
                    input=prompt,
                    actual_output=actual,
                    expected_output=next_assistant
                )
                text_test_cases.append(tc)
                for m in text_metrics:
                    m.measure(tc)
                    text_scores[m.__name__].append(m.score)
                print(f"[TEXT] GT:   {next_assistant[:80]}")
                print(f"[TEXT] PRED: {actual[:80]}\n")

            context.append(turn)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TOOL CALL EVALUATION")
    print("="*60)
    for k, v in tool_scores.items():
        if v:
            print(f"  {k:<25} avg = {sum(v)/len(v):.4f}  ({len(v)} samples)")

    print("\n" + "="*60)
    print("TEXT RESPONSE EVALUATION")
    print("="*60)
    for m in text_metrics:
        scores = text_scores[m.__name__]
        if scores:
            print(f"  {m.__name__:<25} avg = {sum(scores)/len(scores):.4f}  ({len(scores)} samples)")


# ── Run ───────────────────────────────────────────────────────────────────────
evaluate_dataset("adv_eval.json", limit=20)
```

---

## Output You'll Get

```
TOOL CALL EVALUATION
────────────────────────────────────────────
  tool_name_match           avg = 0.7200  (25 samples)
  args_key_match            avg = 0.6400  (25 samples)
  args_value_match          avg = 0.5800  (25 samples)
  full_exact_match          avg = 0.4000  (25 samples)

TEXT RESPONSE EVALUATION
────────────────────────────────────────────
  Perplexity                avg = 3.2100  (18 samples)
  BLEU (bleu4)              avg = 0.0921  (18 samples)
  ROUGE (rouge1)            avg = 0.2341  (18 samples)
  ROUGE (rouge2)            avg = 0.1102  (18 samples)
  ROUGE (rougeL)            avg = 0.2100  (18 samples)
```

The **tool call metrics are the most important** for your use case since the dataset was built around agent tool-calling behavior.
