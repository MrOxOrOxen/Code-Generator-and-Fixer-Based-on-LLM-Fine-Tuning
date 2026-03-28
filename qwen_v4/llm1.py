from qwen_run import qwen
from qwen_command import cmd_llm1

def LLM1(prompts: list[str], qwen_path: str):
    system_msg = {"role": "system", "content": cmd_llm1}
    messages_list = []
    for prompt in prompts:
        messages = [
            system_msg,
            {"role": "user", "content": f"Prompt: {prompt}"}
        ]
        messages_list.append(messages)

    codes = qwen(messages_list, qwen_path)

    return codes