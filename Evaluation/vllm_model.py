import re
import vllm
from config import VLLM_GPU_MEMORY_UTILIZATION,VLLM_TENSOR_PARALLEL_SIZE
class VllmModel:
    def __init__(self,pretrain_path,system_prompt=""):
        self.model_name = pretrain_path
        self.system_prompt = system_prompt
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,max_model_len=16000, enable_lora=False
        )
        self.default_kwargs = { 
            'max_tokens': 500,
            'top_p': 0.8,
            "top_k": 20,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            'stop_token_ids': [151645, 151643],
        }
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side='left'
        self.think_flag=False
    def predict(self,prompt,instruction=None,lora_request=None):
        if instruction is None:
            instruction=self.system_prompt
        messages = [{"role":"system","content":instruction},{"role": "user", "content": prompt}]
        sampling_params = vllm.SamplingParams(**self.default_kwargs)
        response = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params, 
            use_tqdm = False,
            lora_request=lora_request
        )[0]
        output= response.outputs[0].text
        if not self.think_flag:
            return output
        else:
            pattern = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL)
            match = pattern.search(output)
            if not match:
                print(output)
                return output
            #print(query)
            query=match.group(1)
            return query