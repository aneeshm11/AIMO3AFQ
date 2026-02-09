if __name__ == "__main__":

    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    import time
    import json
    import re
    from tqdm import tqdm

    import torch
    from vllm import LLM, SamplingParams


    deadline = time.time() + 14 * 60 * 60

    llm_model_pth = "./models/gpt-oss-120b"

    llm = LLM(
        llm_model_pth,
        max_num_seqs=2048,
        max_model_len=120_000,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        seed=2026,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=1.0,
        min_p = 0.02 , 
        max_tokens=100_000,
    )

    SYSTEM_PROMPT = """
You are an expert olympiad-level mathematics problem composer.

You will be given two problems and their solutions.
You may freely use them for reasoning.

Your task is to create ONE new, much harder problem that fuses ideas from both.

IMPORTANT RULES:
- Do NOT mention Problem 1, Problem 2, their solutions, or any sources.
- Do NOT reference where any idea or hint comes from.
- Do NOT say things like “as before”, “from earlier”, “refer to”, or similar.
- Simply present the fused problem and a clean list of hints.

HINTS:
- Write hints as plain mathematical guidance.
- Do not explain why a hint is valid.
- Do not justify or cite anything.
- Just state the hints directly.
- No need to return the solved answer or solution, strictly only the hints are required

LATEX RULES:
- Use LaTeX only for mathematical expressions.
- Inline LaTeX for simple expressions is preferred.
- Display LaTeX only when necessary.
- Excessive usage of LaTeX should be avoided.

OUTPUT FORMAT (STRICT):

<BOXED_QUESTION>
the new fused problem
</BOXED_QUESTION>

<BOXED_HINTS>
- at most 10 concise bullet-point hints
</BOXED_HINTS>
    """ 

    USER_TEMPLATE_FULL = """
    Problem 1:
    {q1}

    Answer 1:
    {a1}

    Problem 2:
    {q2}

    Answer 2:
    {a2}

    Create the fused problem using the provided information.
    """

    USER_TEMPLATE_TRUNC = """
    Problem 1:
    {q1}

    Answer 1 (solution continues beyond what is shown):
    {a1}
    ... so on.

    Problem 2:
    {q2}

    Answer 2 (solution continues beyond what is shown):
    {a2}
    ... so on.

    Create the fused problem using the provided information.
    """

    from typing import Optional, Tuple

    def extract(text: str) -> Tuple[Optional[str], Optional[str]]:
        def extract_last(tag: str) -> Optional[str]:
            
            pattern = re.compile(
                rf"<{tag}>((?:(?!<{tag}>).)*?)</{tag}>", 
                re.DOTALL | re.IGNORECASE
            )
            
            matches = list(pattern.finditer(text))
            if not matches:
                return None
            return matches[-1].group(1).strip()
    
        last_question = extract_last("BOXED_QUESTION")
        last_hints = extract_last("BOXED_HINTS")
    
        return last_question, last_hints

    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    with open("./generate_questions_putnam.json", "r") as f:
        data = json.load(f)


    data = data[:]
    BATCH = 2000
    results = []
    base_time = time.time()

    pbar = tqdm(chunked(data, BATCH), desc="Generating", unit="batch")

    for batch in pbar:

        if time.time() > deadline:
            print("Deadline reached — stopping safely")
            break

        prompts = []
        meta = []

        MAX_ANS_TOKENS = 10_000

        for x in batch:

            a1_full = x["answer_1"]
            a2_full = x["answer_2"]

            a1_truncated = len(a1_full) > MAX_ANS_TOKENS
            a2_truncated = len(a2_full) > MAX_ANS_TOKENS

            if a1_truncated or a2_truncated:
                user_prompt = USER_TEMPLATE_TRUNC.format(
                    q1=x["question_1"],
                    a1=a1_full[:MAX_ANS_TOKENS],
                    q2=x["question_2"],
                    a2=a2_full[:MAX_ANS_TOKENS],
                )
            else:
                user_prompt = USER_TEMPLATE_FULL.format(
                    q1=x["question_1"],
                    a1=a1_full,
                    q2=x["question_2"],
                    a2=a2_full,
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="high",
            )

            prompts.append(prompt)
            meta.append(x)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        for x, o in zip(meta, outputs):

            text = o.outputs[0].text.strip()

            boxed_q, boxed_h = extract(text)

            results.append({
                "question_1": x["question_1"],
                "question_2": x["question_2"],
                "answer_1": x["answer_1"],
                "answer_2": x["answer_2"],
                "llm_output": text,
                "fused_question": boxed_q,
                "hints": boxed_h,
            })

        elapsed = (time.time() - base_time) / 60
        pbar.set_postfix(min=f"{elapsed:.1f}")

    with open("./fused_questions_putnam.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("TOTAL SAMPLES:", len(results))
    print("TOTAL TIME (min):", (time.time() - base_time) / 60)
