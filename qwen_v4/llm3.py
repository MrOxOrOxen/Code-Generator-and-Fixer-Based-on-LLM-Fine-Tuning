from qwen_command import cmd_llm3_optimize
from qwen_run import qwen
from ast import literal_eval
from itertools import combinations
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import os, traceback
import numpy as np

MAX_ATTEMPT = 10

def LLM3_optimize(
    chosens: list[str],
    codes: list[str],
    base_model_path: str,
    adapter_path: str,
    batch_size=4
):
    messages_list = []

    for chosen, code in zip(chosens, codes):
        for _ in range(MAX_ATTEMPT):
            messages_list.append([
                {"role": "system", "content": cmd_llm3_optimize},
                {"role": "user", "content": f"LLM2 Estimate: {chosen}, code: {code}"}
            ])

    optimized_code_noid = qwen(messages_list, base_model_path, adapter_path, batch_size=4)

    optimized_code_group = []

    index = 0
    for _ in range(len(codes)):
        group = []

        for _ in range(MAX_ATTEMPT):
            optimized_code = optimized_code_noid[index].strip()
            index += 1

            if optimized_code.startswith(("'", '"')) and optimized_code.endswith(("'", '"')):
                optimized_code = optimized_code[1:-1]

            if optimized_code not in group:
                group.append(optimized_code)
            
        if len(group) > 3:
            group = group[:3]
        elif len(optimized_code_group) < 3:
            group += [group[-1]] * (3-len(group))
        optimized_code_group.append(group)

    return optimized_code_group
        


def LLM3_exec(code: str, test_case: str):
    try:
        namespace = {}
        exec(code, namespace)
        funcs = [v for v in namespace.values() if callable(v)]
        func = funcs[0]
        args = literal_eval(test_case)
        if isinstance(args, dict):
            func(**args)
        else:
            if isinstance(args, tuple):
                func(*args)
            else:
                func(args)
        return {"error": None, "line": -1}
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        line = tb[-1].lineno if tb else e.lineno if hasattr(e, 'lineno') else -1
        return {"error": type(e).__name__, "line": line}

def LLM3_filter(chosen: dict, optimized_code_group: list, test_case: list, problem_desc: str):
    score = [0] * len(optimized_code_group)
    dpo_group = []
    for i, optimized_code in enumerate(optimized_code_group):
        for value in test_case:
            exec_result = LLM3_exec(optimized_code, value)
            if exec_result == chosen:
                score[i] += 1
            elif exec_result["error"] == chosen["error"]:
                score[i] += 0.5
            elif exec_result["line"] == chosen["line"]:
                score[i] += 0.5
            else:
                score[i] -= 0.5
    score = [max(s, 0) for s in score]
    for i, j in combinations(range(len(score)), 2):
        dpo_pair = {}
        excellence_level = score[i] / score[j] if score[j] != 0 else np.inf
        if excellence_level >= 1:
            c = optimized_code_group[i]
            r = optimized_code_group[j]
        else:
            c = optimized_code_group[j]
            r = optimized_code_group[i]
            excellence_level = 1 / excellence_level if excellence_level != 0 else np.inf
        dpo_pair["chosen"] = c
        dpo_pair["rejected"] = r
        dpo_pair["excellence_level"] = excellence_level
        dpo_pair["problem_description"] = problem_desc
        dpo_group.append(dpo_pair)
    max_index = score.index(max(score))
    optimized_code = optimized_code_group[max_index]
    return dpo_group, optimized_code

def LLM3_DPO(
    dpo_groups: list,
    base_model_path: str,
    prev_adapter_path: str | None,
    output_adapter_path: str
):
    prompts, chosens, rejecteds = [], [], []
    for dpo_group in dpo_groups:
        problem_desc = dpo_group["problem_description"]
        chosen = dpo_group["chosen"]
        rejected = dpo_group["rejected"]
        if chosen == rejected:
            continue
        prompt = f"""You are an expert Python programmer.
Solve the following problem.
### Problem:
{problem_desc}
### Python Solution""".strip()
        prompts.append(prompt)
        chosens.append(str(chosen))
        rejecteds.append(str(rejected))
    dpo_dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if prev_adapter_path and os.path.exists(prev_adapter_path):
        print(f"### Loading previous LoRA adapter: {prev_adapter_path} ###")
        model = PeftModel.from_pretrained(base_model, prev_adapter_path)
    else:
        print("### Initializing new LoRA ###")
        model = get_peft_model(base_model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_adapter_path, "training_args"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none"
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=1024,
        max_prompt_length=256
    )
    trainer.train()
    # model_output_dir = output_path + "/model"
    # trainer.save_model(model_output_dir)

    os.makedirs(output_adapter_path, exist_ok=True)
    model.save_pretrained(output_adapter_path)
    return model