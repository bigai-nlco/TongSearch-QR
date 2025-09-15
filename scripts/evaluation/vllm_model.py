import re
from typing import Optional, Dict, Any
import vllm


class VllmModel:
    """
    Wrapper for vLLM LLM model with optional system prompt and <think> parsing.
    """
    
    DEFAULT_SAMPLING_PARAMS = {
        'max_tokens': 500,
        'top_p': 0.8,
        'top_k': 20,
        'temperature': 0.7,
        'repetition_penalty': 1.05,
        'stop_token_ids': [151645, 151643],
    }

    THINK_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL)

    def __init__(self, pretrain_path: str, system_prompt: str = "") -> None:
        self.model_name = pretrain_path
        self.system_prompt = system_prompt
        self.think_flag = False

        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,
            max_model_len=16000,
            enable_lora=False
        )

        self.default_kwargs = self.DEFAULT_SAMPLING_PARAMS.copy()
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = 'left'

    def predict(self, prompt: str, instruction: Optional[str] = None, lora_request: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate model output given a prompt and optional instruction.

        Args:
            prompt: User query or input text.
            instruction: Optional system-level instruction; defaults to self.system_prompt.
            lora_request: Optional LoRA adaptation request.

        Returns:
            Generated text. If think_flag is True, returns the content inside <answer> tags.
        """
        instruction = instruction or self.system_prompt
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]

        sampling_params = vllm.SamplingParams(**self.default_kwargs)
        response = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request
        )[0]

        output_text = response.outputs[0].text
        return self._parse_think_output(output_text) if self.think_flag else output_text

    def _parse_think_output(self, text: str) -> str:
        """
        Extract text inside <answer> tags if think_flag is True.

        Args:
            text: Raw model output.

        Returns:
            Extracted answer or raw text if pattern not found.
        """
        match = self.THINK_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        print("Think pattern not found in output:", text)
        return text