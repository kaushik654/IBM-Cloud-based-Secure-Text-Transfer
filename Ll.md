import re

def generate(prompt: str, max_new_tokens: int = 512) -> str:  # ← increase tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,      # ← 512 minimum
            eos_token_id=stop_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated = out[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    decoded = decoded.replace("<|im_end|>", "").strip()
    return decoded


def extract_tool_call(raw_output: str) -> str:
    """
    Strip <think>...</think> block and return only the tool call JSON.
    """
    # Remove <think>...</think> block (including multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    return cleaned


def eval_tool_call(predicted: str, ground_truth: str) -> dict:
    results = {
        "tool_name_match": 0,
        "args_key_match": 0,
        "args_value_match": 0,
        "full_exact_match": 0,
    }
    try:
        # ← Strip think tags before parsing
        predicted_clean = extract_tool_call(predicted)

        pred_list = json.loads(predicted_clean)
        gt_list   = json.loads(ground_truth)

        pred_tool = pred_list[0]
        gt_tool   = gt_list[0]

        if pred_tool.get("name") == gt_tool.get("name"):
            results["tool_name_match"] = 1

        pred_args = pred_tool.get("arguments", {})
        gt_args   = gt_tool.get("arguments", {})

        if set(pred_args.keys()) == set(gt_args.keys()):
            results["args_key_match"] = 1

        matches = sum(1 for k in gt_args if pred_args.get(k) == gt_args[k])
        results["args_value_match"] = matches / len(gt_args) if gt_args else 0

        if pred_tool == gt_tool:
            results["full_exact_match"] = 1

    except json.JSONDecodeError as e:
        # Print what failed to parse for debugging
        print(f"  [PARSE ERROR] {e}")
        print(f"  [RAW OUTPUT] {predicted[:150]}")

    return results
