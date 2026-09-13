from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import  PeftModel
import os 
import torch
from router_prompt import ROUTER_PROMPT_TEMPLATE

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_LORA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),'output','router_lora')

base_infer = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
router_infer = PeftModel.from_pretrained(base_infer, OUTPUT_LORA_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def _classify_lora(user_text) ->int:
    prompt = ROUTER_PROMPT_TEMPLATE.format(query=user_text)
    inputs = tok(prompt, return_tensors='pt')
    outputs = router_infer.generate(**inputs,max_new_tokens=8)
    n = inputs['input_ids'].shape[1]
    text_out = tok.decode(outputs[0][n:],skip_special_tokens=True)
    
    for ch in text_out:
        if ch in '0123':
            return int(ch)
    return 0