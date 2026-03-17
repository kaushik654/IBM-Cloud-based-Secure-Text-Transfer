from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL_PATH = "./deepeval/Qwen3-0.6B"
LORA_PATH = "./deepeval/qwen_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()  # optional but recommended

model.eval()
