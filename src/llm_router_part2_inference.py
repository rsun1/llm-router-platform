from dotenv import load_dotenv
import os 
import yaml
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from openai import OpenAI
from anthropic import Anthropic
from utils.token_tools import count_messages_tokens, check_context_valid, predict_completion_tokens
from router_prompt import ANSWER_PROMPT_TEMPLATE
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from functools import lru_cache
import json
from llm_router_part6_adapters import load_registry
    

OLLAMA_MODEL_MAP = {
    'mistral-7b': 'mistral',
    'llama-3.1-70b': 'llama3.1',
}
ANTHROPIC_MODEL_MAP = {
    'claude-3.5-sonnet': 'claude-sonnet-4-5-20250929',
}

TOKEN_MODEL_MAP = {
    'mistral-7b': 'mistral:7b',
    'llama-3.1-70b': 'llama3.1:8b',
}
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

load_dotenv()
configpath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config')

def load_config():
    config = os.path.join(configpath, 'config.yaml')
    with open(config) as f:
        return yaml.safe_load(f)

def call_ollama(selected_model, user_text):
    ollama_name = OLLAMA_MODEL_MAP[selected_model]
    llm = ChatOllama(model=ollama_name,
                     temperature=0
                     )
    res = llm.invoke([HumanMessage(content=user_text)])
    return res.content

def call_openai(model_info, selected_model, user_text):
    api_key = os.environ.get(model_info['api_key_env']) 
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user","content":user_text}]
    )
    return response.choices[0].message.content

@lru_cache(maxsize=1)
def _get_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base_model, os.path.join(os.path.dirname(os.path.dirname(__file__)),'output','answer_code_lora'), adapter_name='code')
    peft_model.load_adapter(os.path.join(os.path.dirname(os.path.dirname(__file__)),'output','answer_legal_lora'),adapter_name='legal')
    return tok, peft_model

def _generate_local(user_text, domain=None):
    tok, model = _get_model()
    prompt = ANSWER_PROMPT_TEMPLATE.format(query=user_text)
    inputs = tok(prompt, return_tensors='pt')
    
    if domain in model.peft_config:
        model.set_adapter(domain)
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    else:
        with model.disable_adapter():
            outputs = model.generate(**inputs,max_new_tokens=150, do_sample=False)
    n=inputs['input_ids'].shape[1]
    return tok.decode(outputs[0][n:],skip_special_tokens=True).strip()

def call_local_peft(user_text, adapter_id = None):
    domain = None 
    if adapter_id is not None:
        registry = load_registry()
        record = registry.get(adapter_id)
        if record is not None:
            domain = record['domain']
        else:
            domain =None
    return _generate_local(user_text, domain)
    
def call_anthropic(model_info, selected_model, user_text):
    api_key = os.environ.get(model_info['api_key_env'])
    client = Anthropic(api_key=api_key)
    anthropic_name = ANTHROPIC_MODEL_MAP[selected_model]
    response = client.messages.create(
        model=anthropic_name,
        max_tokens=2048,
        messages=[{"role":"user", "content":user_text}]
    )
    return response.content[0].text 

def check_cost(selected_model, user_tier, input_tokens, token_model_name):
    config = load_config()
    budget = config["policies"]["cost_budgets"][user_tier]
    costs_per_token = config["router"]["models"][selected_model]["cost_per_token"]
    predict_result = predict_completion_tokens(token_model_name, input_tokens)
    predict_output_tokens = predict_result.predicted_tokens 
    
    total_tokens = input_tokens + predict_output_tokens
    estimated_costs = total_tokens * costs_per_token
    
    return estimated_costs <= budget
    

def call_model(selected_model, user_text, user_tier,adapter_id=None):
    config = load_config()
    model_info = config['router']['models'][selected_model]
    provider = model_info['provider']
    
    token_model_name = TOKEN_MODEL_MAP.get(selected_model, selected_model)
    input_tokens = count_messages_tokens([{"role":"user", "content": user_text}], token_model_name)
    is_valid = check_context_valid(token_model_name, input_tokens, 2048)
    
    is_within_budget = check_cost(selected_model, user_tier, input_tokens,token_model_name)
    
    if not is_valid:
        raise Exception("Input too long, exceeded the model's context window.")
    elif not is_within_budget:
        raise Exception("Costs exceeding budget.")
    else:
        if provider == 'vllm':
            return call_ollama(selected_model, user_text)
        elif provider == 'openai':
            return call_openai(model_info, selected_model, user_text)
        elif provider == 'anthropic':
            return call_anthropic(model_info,selected_model, user_text)
        elif provider == 'local_peft':
            return call_local_peft(user_text, adapter_id)
        else:
            raise Exception(f"Unknown provider: {provider}")
        
if __name__ == '__main__':
    #print(call_model('mistral-7b','hi'))
    #print(call_model('gpt-4-turbo','hi'))
    #print(call_model('claude-3.5-sonnet','hello'))
    # print(call_model('llama-3.1-70b', 'hi','free')) 
    # q = 'Can my employer reduce my salary without telling me first?'
    # print('--- legal ---')
    # print(_generate_local(q, 'legal'))
    # print('--- 无适配器 ---')
    # print(_generate_local(q, None))
    # print(call_local_peft(q, None))                    # 纯基座
    # print(call_local_peft(q, 'mistral-7b-legal-v1'))   # registry 里有 → domain=legal → 挂法律适配器
    # print(call_local_peft(q, 'no-such-adapter'))
    pass