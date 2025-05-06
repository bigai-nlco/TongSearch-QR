import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
VLLM_TENSOR_PARALLEL_SIZE = int(torch.cuda.device_count()) # 这里改成实际的卡的数量
print("GPU Num:",VLLM_TENSOR_PARALLEL_SIZE)
VLLM_GPU_MEMORY_UTILIZATION = 0.5 #这里设置每块卡最大用多大比例的显存
import sys
TONGSEARCH_REASONER_PATH=sys.argv[1]
BRIGHT_DATASET_PATH=sys.argv[2]
USE_THINK=False