#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH='Qwen/Qwen2.5-1.5B-Instruct'
trl vllm-serve --model $MODEL_PATH --max_model_len 9000 --port 8000