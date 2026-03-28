This project is based on Qwen3-4B model ([Qwen3-4B Hugging Face Website](https://huggingface.co/Qwen/Qwen3-4B)).

Qwen3-4B requires transformers version >= 4.51.0. For the folder Qwen3-4B, please visit the website above.

# Code structure

![code_structure](/src/structure.png)

# Qwen_v1
*Last updated on Feb 4, 2026*

## Contributions

Build **a single series structure without feedback**.

## Functions

Achieved: Function LLM1, LLM2_estimate and LLM3_optimize.

LLM1 receives user input and generates a code with errors.

LLM2_estimate estimates the possible error of the code and generates a possible error type.

LLM3_optimize modified the code based on the predicted error.

# Qwen_v2
*Last updated on March 19, 2026*

**This version has not been compiled.**

## Contributions

**Copy and paste** the aforementioned serial structure into a parallel structure (i.e. Circ 1), where the first channel requires a model to be loaded, and other channels use the loaded model. **Circ 1 is not processed in parallel.**

Complete the circular logic and HumanEval test module of the subsequent Circ.

Implement DPO for model training. In this version, **a new DPO model will be saved everytime when the model is trained**.

## Functions

Updated: Function LLM2_estimate, LLM3_optimize.

Achieved: Function LLM2_filter, LLM2_DPO, LLM3_filter, LLM3_DPO

LLM2_estimate will generate two different possible error types and lines now.

LLM2_filter generates a chosen and a rejected prediction. The prediction pair is used for DPO training.

LLM2_DPO will train and generate a new LLM2 model.

LLM3_optimize will generate three different fixed codes, and compare the modifying accuracy by pairwise code comparison.

LLM3_filter generates a chosen and a rejected code. The code pair is used for DPO training.

LLM3_DPO will train and generate a new LLM3 model.

# Qwen_v3
*Last updated on March 20, 2026*

**This version has not been compiled.**

## Contributions

Circ 1 is processed in parallel now through batch. Qwen function is also updated. Thus, the pressure on the GPU decreases sharply.

## Functions

Updated: LLM2_estimated, LLM2_DPO, LLM3_optimize, LLM3_DPO

Batch is used in the functions above.

# Qwen_v4
*Last updated on March 23, 2026*

## Contributions

LoRA is implemented when the model is being trained by DPO. In this version, **a new DPO model will not be saved**. A LoRA file will be saved instead.

When loading model, a base model and a LoRA file will be loaded. This is equivalent to the new model being loaded.

By storing LoRA files only, the storage space of the server is freed.

## Functions

Updated: LLM2_DPO, LLM3_DPO

