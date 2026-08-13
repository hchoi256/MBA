import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaModel:
    def __init__(self, model_path="/workspace/mistral-7b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.model.eval()
        self.cache_dict = {}

    def save_cache(self):
        pass

    def generate(self, prompt, **kwargs):
        system_msg = "You are a factual information extraction assistant. Break the following text into independent, atomic facts."
        formatted_prompt = f"<s>[INST] {system_msg}\n\nText: {prompt} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), None