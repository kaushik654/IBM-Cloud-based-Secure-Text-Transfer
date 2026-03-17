This is very important information. Now I can see the exact prompt template used during training. This means for proper evaluation, your generate() function must use the same template, otherwise the model won't produce correct outputs.
What This Template Tells Us
name = "qwen3_legacy_obs"
system = "You are AIM, created by Samsung Research America..."
tool_format = "qwen"
template_class = ReasoningTemplate
The prompt format is:
<|im_start|>system
You are AIM, created by Samsung Research America...
<|im_end|>
<|im_start|>user
{user message + tools}
<|im_end|>
<|im_start|>assistant
{tool call or text}
<|im_end|>
<|im_start|>observation
{tool result}
<|im_end|>
<|im_start|>assistant
{follow-up response}
Updated build_prompt() and generate()
Replace your existing functions with these:
import json

SYSTEM_PROMPT = "You are AIM, created by Samsung Research America. You are a helpful assistant who must address user queries by calling on of the provided tools, asking for clarification, or conversing with the user."

def format_tools_qwen(tools: list) -> str:
    """Format tools in Qwen tool_format style."""
    return json.dumps(tools, indent=2)

def build_prompt(tools: list, conversations: list) -> str:
    """
    Build prompt exactly matching qwen3_legacy_obs template used in training.
    """
    tool_str = format_tools_qwen(tools)

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nTools:\n{tool_str}<|im_end|>\n"

    for turn in conversations:
        role  = turn["from"]
        value = turn["value"]

        if role == "user":
            prompt += f"<|im_start|>user\n{value}<|im_end|>\n<|im_start|>assistant\n"

        elif role == "assistant":
            prompt += f"{value}<|im_end|>\n"

        elif role == "observation":
            prompt += f"<|im_start|>observation\n{value}<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate using the exact same stop words as training: <|im_end|>"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # get the token id for <|im_end|> as stop token
    stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop_token_id,      # ← matches stop_words=["<|im_end|>"]
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,                 # greedy — consistent with eval
        )

    generated = out[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False)

    # strip trailing <|im_end|> if present
    decoded = decoded.replace("<|im_end|>", "").strip()
    return decoded
Updated evaluate_dataset() call
The only change needed is how you build the context before each user turn:
for i, turn in enumerate(conversations):
    if turn["from"] != "user":
        context.append(turn)
        continue

    # Pass everything before this user turn as context
    prompt = build_prompt(tools, context + [turn])
    actual = generate(prompt)

    # ... rest of eval logic unchanged
    context.append(turn)
Why This Matters
Without correct template
With correct template
Model sees raw text, no special tokens
Model sees <|im_start|> / <|im_end|> it was trained on
Model doesn't know when to stop
stop_words=["<|im_end|>"] used correctly
Tool calls may not be formatted as JSON
tool_format="qwen" respected
Low scores due to template mismatch
Scores reflect true model capability
Without this fix your BLEU/ROUGE/tool accuracy scores would be artificially low because the model is being prompted differently from how it was trained.
