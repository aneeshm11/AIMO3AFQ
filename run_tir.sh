set -euo pipefail
set -ex
# this will run one after another

python solve_tir.py \
  --model-path ../models/gpt-oss-120b \
  --port 10100 \
  --tensor-parallel 2 \
  --gpu-mem-util 0.85 \
  --max-model-len 100000 \
  --context-len 100000 \
  --max-output-tokens 90000 \
  --temperature 1.0 \
  --top-p 1.0 \
  --min-p 0.02 \
  --seed 42 \
  --max-iter 60 \
  --request-timeout-s 3600 \
  --cutoff-hours 30 \
  --max-workers 128 \
  --dataset-name ../datasets/numina_cot \
  --splits train \
  --source-values olympiads \
  --start-index 0 \
  --end-index 80000 \
  --out-json ./runs/numina_cot_0_80000.json \
  --prompt "You are an expert olympiad-level mathematics problem solver. Reason step by step and use the python tool to solve the math problem. Return the final answer in \\boxed{}."


python solve_tir.py \
  --model-path ../models/gpt-oss-120b \
  --port 10100 \
  --tensor-parallel 2 \
  --gpu-mem-util 0.85 \
  --max-model-len 100000 \
  --context-len 100000 \
  --max-output-tokens 90000 \
  --temperature 1.0 \
  --top-p 1.0 \
  --min-p 0.02 \
  --seed 42 \
  --max-iter 60 \
  --request-timeout-s 3600 \
  --cutoff-hours 30 \
  --max-workers 128 \
  --dataset-name ../datasets/numina_tir \
  --splits train \
  --out-json ./runs/numina_tir.json \
  --prompt "You are an expert olympiad-level mathematics problem solver. Reason step by step and use the python tool to solve the math problem. Return the final answer in \\boxed{}."

python solve_tir.py \
  --model-path ../models/gpt-oss-120b \
  --port 10100 \
  --tensor-parallel 2 \
  --gpu-mem-util 0.85 \
  --max-model-len 100000 \
  --context-len 100000 \
  --max-output-tokens 90000 \
  --temperature 1.0 \
  --top-p 1.0 \
  --min-p 0.02 \
  --seed 42 \
  --max-iter 60 \
  --request-timeout-s 3600 \
  --cutoff-hours 30 \
  --max-workers 128 \
  --dataset-name ../datasets/openmath_tir.json \
  --out-json ./runs/openmath_tir.json \
  --prompt "You are an expert olympiad-level mathematics problem solver. Reason step by step and use the python tool to solve the math problem. Return the final answer in \\boxed{}."
