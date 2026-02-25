Yes — GGUF → SafeTensors conversion is possible, but it is not a direct file conversion (like renaming or simple format swap).
You must reconstruct the Hugging Face model and then save it as SafeTensors.


---

🔁 Why direct conversion is not possible

GGUF → optimized for llama.cpp inference

SafeTensors → standard for PyTorch / Hugging Face

GGUF:

weights are quantized & packed

architecture info is partially baked


SafeTensors:

expects full-precision (or HF-compatible quantized) tensors with HF config



So the pipeline is:

> GGUF → load with llama.cpp tools → convert to HF → save as SafeTensors




---

✅ Method 1 — Using llama.cpp converter (recommended)

1️⃣ Clone llama.cpp

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt


---

2️⃣ Convert GGUF → HF

python convert_hf.py \
  --outfile model_hf \
  --outtype f16 \
  path/to/model.gguf

This gives you:

model_hf/
 ├── config.json
 ├── pytorch_model.bin
 ├── tokenizer.model / tokenizer.json


---

3️⃣ Convert HF → SafeTensors

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("model_hf")
model.save_pretrained("model_safetensors", safe_serialization=True)

✅ Now you get:

model.safetensors


---

⚠️ VERY IMPORTANT (Quantization limitation)

If your GGUF is:

GGUF type	Result

Q2 / Q3 / Q4 / Q5 / Q8	❌ Cannot recover original FP weights
F16 / BF16	✅ Works correctly


So:

👉 Quantized GGUF → you will get a quantized HF model, not full precision
👉 Quality depends on the quantization


---

🧠 When this is useful (your use cases)

Since you’re working with:

ONNX

HF

quantized NER / LLM pipelines


You need SafeTensors for:

Transformers loading

Optimum / ONNX export

TensorRT / OpenVINO


So this pipeline is the correct path.


---

🚀 One-line automated script

python llama.cpp/convert_hf.py --outfile tmp --outtype f16 model.gguf
python - <<EOF
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("tmp")
m.save_pretrained("final", safe_serialization=True)
EOF


---

❗ Common errors

Rope scaling mismatch → config fix needed

Tokenizer missing → copy from base model

Architecture not supported → only LLaMA/Mistral/Qwen/Phi supported



---

⭐ If you want — I can give you

A perfect command for your exact GGUF model

Tell me:

model name

quantization type (Q4_K_M, Q8_0, F16, etc.)


and I’ll give you a working end-to-end script for your setup.
