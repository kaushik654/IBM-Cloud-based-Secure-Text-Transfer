def is_tool_call(value: str) -> bool:
    """Check if assistant turn is a tool call, ignoring think tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        return isinstance(parsed, list) and len(parsed) > 0 and "name" in parsed[0]
    except:
        return False
```

---

## Expected Result After Fix
```
[TOOL] GT:   [{"name":"manage_software_updates","arguments":{"action":"install_updates"}}]
[TOOL] PRED: [{"name":"manage_software_updates","arguments":{"action":"install_updates"}}]
[TOOL] Scores: {'tool_name_match': 1, 'args_key_match': 1, 'args_value_match': 1.0, 'full_exact_match': 1}
