if __name__ == "__main__":

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    import json
    import time
    import re
    from tqdm import tqdm
    from typing import Optional

    import torch
    from vllm import LLM, SamplingParams
    from transformers import set_seed
    set_seed(42)

    MODEL_PATH = "./models/gpt-oss-120b"
    INPUT_JSON = "../nvidia_cot_large.json"
    OUTPUT_JSON = "./results_openmath_cot.json"

    BATCH_SIZE = 2000
    DEADLINE = time.time() + 34 * 60 * 60

    BOX_RE = re.compile(
        r"\\boxed\s*\{\s*(-?\d+)\s*\}", re.IGNORECASE
    )

    def extract_boxed_int(text: str) -> Optional[int]:
        matches = BOX_RE.findall(text)
        for m in reversed(matches):
            try:
                return int(m)
            except:
                pass
        return None

    SYSTEM_PROMPT = (
        "You are an expert olympiad-level mathematics problem solver.\n"
        "Solve the given problem.\n"
        "Return the final answer in boxed{}."
    )

    llm = LLM(
        MODEL_PATH,
        max_num_seqs=2048,
        max_model_len=120_000,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        seed=42,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=1.0,
        min_p = 0.02 , 
        max_tokens=100_000,
    )

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    
    # data = data[:]

    results = []

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Solving"):

        if time.time() > DEADLINE:
            print("⏹ Deadline reached")
            break

        batch = data[i : i + BATCH_SIZE]
        prompts = []

        for x in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ]

            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort = "high"
            )

            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        for x, o in zip(batch, outputs):
            text = o.outputs[0].text.strip()
            pred = extract_boxed_int(text)

            gt = int(x["answer"])
            correct = (pred == gt)

            results.append({
                "question": x["question"],
                "answer": gt,
                "pass_rate_72b_tir": x["pass_rate_72b_tir"],
                "model_output": text,
                "predicted_answer": pred,
                "correct": correct,
            })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("TOTAL:", len(results))
