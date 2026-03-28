from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

_model_cache = {}

def load_model(base_model_path: str, adapter_path: str = None):
    key = (base_model_path, adapter_path)
    if key in _model_cache:
        return _model_cache[key]

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"### Loading LoRA adapter from: {adapter_path} ###")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        if adapter_path:
            print(f"### Adapter path {adapter_path} not found, using base model ###")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()
    _model_cache[key] = (model, tokenizer)
    return model, tokenizer

def qwen(messages_list, base_model_path, adapter_path=None, batch_size=4):
    model, tokenizer = load_model(base_model_path, adapter_path)

    all_outputs = []

    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i:i+batch_size]

        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in batch_messages
        ]

        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        for j in range(len(texts)):
            input_len = model_inputs["attention_mask"][j].sum().item()
            output_ids = generated_ids[j][input_len:].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            code = tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True
            ).strip("\n")

            all_outputs.append(code)

    return all_outputs