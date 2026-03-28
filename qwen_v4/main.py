from human_eval.data import read_problems
from llm1 import LLM1
from llm2 import LLM2_estimate, LLM2_filter, LLM2_DPO
from llm3 import LLM3_optimize, LLM3_filter, LLM3_DPO
from llm_path import *
import re, traceback

MAX_CIRC = 150
MAX_CYCLE = 50

circ = 0
cycle = 0
problems = read_problems()

test_cases = {}
ids = []
prompts = []

base_model_path = "/home/yjx/Qwen3-4B"

# Initialize
print("### Initializing ###")
for circ in range(MAX_CIRC):
    id = f"HumanEval/{circ}"
    ids.append(id)
    problem = problems[id]
    test = problem['test']

    pattern = r'candidate\((.*?)\)\s*=='
    matches = re.findall(pattern, test, re.DOTALL)
    cases = []
    for m in matches:
        case = re.sub(r'\s+', '', m.strip())
        cases.append(case)
    test_cases[id] = cases
    test_cases_list = [test_cases[id] for id in ids]

    prompt = problems[id]['prompt']
    prompts.append(prompt)

# main cycle
for cycle in range(MAX_CYCLE):
    print(f"### CYCLE {cycle} ###")
    # LLM1
    if cycle == 0:
        print("### LLM1 Generating Code ###")
        codes_noid = LLM1(prompts, llm1_qwen_path)
        codes = {id: code for id, code in zip(ids, codes_noid)}

    # LLM2
    codes_list = [codes[id] for id in ids]

    print("### LLM2 Estimation Generating ###")
    adapter_path = f"/yjx/home/lora/llm2_v{cycle}" if cycle > 0 else None
    estimation_pairs = LLM2_estimate(codes_list, base_model_path, adapter_path, batch_size=4)

    print("### LLM2 Filtering ###")
    dpo_pairs = []
    for id, code, est_pair, test_case in zip(ids, codes_list, estimation_pairs, test_cases_list):
        dpo_pair = LLM2_filter(code, est_pair, test_case)
        dpo_pairs.append(dpo_pair)

    HE_ready = True

    for dpo_pair in dpo_pairs:
        if dpo_pair["chosen"] != {"error": None, "line": -1}:
            HE_ready = False
            break

    if not HE_ready:
        prev_adapter = f"/yjx/home/lora/llm2_v{cycle}" if cycle > 0 else None
        new_adapter = f"/yjx/home/lora/llm2_v{cycle+1}"
        llm2_model = LLM2_DPO(dpo_pairs, base_model_path, prev_adapter, new_adapter)

        # LLM3
        dpo_groups = {}

        chosens_list = [pair["chosen"] for pair in dpo_pairs]

        adapter_path = f"/yjx/home/lora/llm3_v{cycle}" if cycle > 0 else None
        optimized_groups = LLM3_optimize(chosens_list, codes_list, base_model_path, adapter_path, batch_size=4)

        for i, id in enumerate(ids):
            optimized_code_group = optimized_groups[i]
            dpo_group, optimized_code = LLM3_filter(
                chosens_list[i],
                optimized_code_group,
                test_cases_list[i],
                prompts[i]
            )
            dpo_groups[id] = dpo_group
            codes[id] = optimized_code

        prev_adapter = f"/yjx/home/lora/llm3_v{cycle}" if cycle > 0 else None
        new_adapter = f"/yjx/home/lora/llm3_v{cycle+1}"
        llm3_model = LLM3_DPO(list(dpo_groups.values()), base_model_path, prev_adapter, new_adapter)

    elif HE_ready:
        # HumanEval test
        all_pass = True
        for circ in range(MAX_CIRC):
            id = f"HumanEval/{circ}"
            prompt = problems[id]['prompt']
            test = problems[id]['test']

            test_code = codes[id]

            test_code = test.replace('    """', '    """'+test_code)

            total_count = len(re.findall(r'assert\s+candidate', test))
            pass_count = 0

            try:
                local_namespace = {}
                exec(test_code, local_namespace)
                intersperse_func = local_namespace["intersperse"]
                check_func = local_namespace["check"]

                check_func(intersperse_func)

                print(f"Pass {pass_count}/{total_count}")

            except AssertionError:
                all_pass = False
                print(f"AssertionError {pass_count}/{total_count}")

            except Exception as e:
                all_pass = False
                tb = traceback.extract_tb(e.__traceback__)
                line = tb[-1].lineno if tb else e.lineno if hasattr(e, 'lineno') else -1

                print(f"Error Type: {type(e).__name__}, line: {line}")

        if all_pass:
            print("All pass. Training done.")
            break
        else:
            continue

