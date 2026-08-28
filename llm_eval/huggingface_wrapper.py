from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import os
from pathlib import Path

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def _is_model_cached(model_name: str) -> bool:
    """
    Check if a model is already fully cached locally.
    HF cache structure: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/
    Returns True if at least one snapshot directory exists and is non-empty.
    """
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    # Convert 'google/gemma-4-E4B-it' -> 'models--google--gemma-4-E4B-it'
    model_dir_name = "models--" + model_name.replace("/", "--")
    snapshots_dir  = cache_dir / model_dir_name / "snapshots"

    if not snapshots_dir.exists():
        return False

    # Check that at least one snapshot folder contains files
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir() and any(snapshot.iterdir()):
            return True

    return False


class HFLocalModel:

    def __init__(self, model_name: str):
        if _is_model_cached(model_name):
            print(f"Model '{model_name}' found in cache — enabling offline mode.")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"]  = "1"
        else:
            print(f"Model '{model_name}' not cached — running in online mode for initial download.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.supports_thinking = self._check_thinking_support()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
        )
        self.model = torch.compile(self.model)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE",  None)



    def _check_thinking_support(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            return True
        except TypeError:
            return False


    def generate(self, prompts: list[str], enable_thinking: bool = False):
        
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.supports_thinking:
            template_kwargs["enable_thinking"] = enable_thinking

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                **template_kwargs
            )
            for p in prompts
        ]

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        print(f"torch.cuda.memory_allocated(): {torch.cuda.memory_allocated()}", flush=True)

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return decoded