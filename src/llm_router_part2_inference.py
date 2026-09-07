from dotenv import load_dotenv
import os 
import yaml
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from openai import OpenAI
from anthropic import Anthropic
from utils.token_tools import count_messages_tokens, check_context_valid, predict_completion_tokens

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
    

def call_model(selected_model, user_text, user_tier):
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

if __name__ == '__main__':
    #print(call_model('mistral-7b','hi'))
    #print(call_model('gpt-4-turbo','hi'))
    #print(call_model('claude-3.5-sonnet','hello'))
    print(call_model('llama-3.1-70b', 'hi','free')) 