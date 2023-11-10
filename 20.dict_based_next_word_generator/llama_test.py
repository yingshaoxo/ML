import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_3b_v2'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

context = ""
while True:
    prompt = input("What you want to say? ")
    context += "\n" + prompt
    context = context[-2000:]
    input_ids = tokenizer("You are a helpful assistant.\n" + context, return_tensors="pt", truncation=True, max_length=2048).input_ids

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=32
    )
    print(tokenizer.decode(generation_output[0]))
