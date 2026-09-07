import os 
import yaml 
from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage

configpath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config')

def load_config():
    config = os.path.join(configpath, 'config.yaml')
    with open(config) as f:
        return yaml.safe_load(f)

TIER_MODELS = {
    'free': ['mistral-7b'],
    'premium': ['gpt-4-turbo','claude-3.5-sonnet']
}
ALL_MODELS = ['gpt-4-turbo','claude-3.5-sonnet','mistral-7b','llama-3.1-70b']

TYPE_MODELS = {
    'code_generation': ['gpt-4-turbo', 'mistral-7b'],
    'analysis': ['claude-3.5-sonnet','llama-3.1-70b'],
}

def count_tokens(text):
    n_tokens = len(text)//4 
    return n_tokens

llm = ChatOllama(
    model='llama3.1',
    temperature=0
)
def classifyChat(user_text):
    prompt = f'''
    Two outputs. First, classify user intent type in: {user_text} and output the corresponding 
    category from one of the following: "code_generation","analysis","general",
    "summarization". 
    Second, classify the domain that {user_text} belongs to, and output the corresponding
    domain from one of the following: "code","customer_service","general". 
    Reply with the specific format:
    Type: classified user intent\nDomain: classified domain
    '''

    res = llm.invoke([HumanMessage(content=prompt)])
    pos = res.content.find("Domain")
    first_part_query_type = res.content[:pos].strip().lower().strip('".') 
    second_part_domain_type = res.content[pos:].strip().lower().strip('".')     
    
    classified = ['code_generation','analysis','general','summarization']
    domain = ['code','customer_service','general']
    result=['general','general']
    
    for c in classified:
        if c in first_part_query_type:
            result[0] = c

    for d in domain:
        if d in second_part_domain_type:
            result[1] = d 
    return result

def text_input(user_text,user_tier):
    query_type, domain = classifyChat(user_text)
    token_count = count_tokens(user_text)
    return query_type, domain, user_tier, token_count

def select_model(query_type, user_tier,token_count):
    allowed_models = TIER_MODELS.get(user_tier, ALL_MODELS)
    
    if query_type == 'code_generation':
        preferred_models = TYPE_MODELS['code_generation']
    elif query_type == 'analysis' and token_count > 50000:
        preferred_models = TYPE_MODELS['analysis']
    else:
        preferred_models = ALL_MODELS
        
    candidates = []
    for model in allowed_models:
        if model in preferred_models:
            candidates.append(model)
    if not candidates:
        candidates = allowed_models
        
        
    config = load_config()
    model_priority = {}
    for model_name, model_info in config['router']['models'].items():
        model_priority[model_name] = model_info['priority']
    best_model = min(candidates, key=lambda m: model_priority[m])
    return best_model

def pick_fallback_model(selected_model, user_tier):
    allowed_models = TIER_MODELS.get(user_tier, ALL_MODELS)
    candidates = [ x for x in allowed_models if x != selected_model]
    if not candidates:
        return None 
    config = load_config()
    model_costs = {}
    for model_name, model_info in config['router']['models'].items():
        model_costs[model_name] = model_info['cost_per_token']
    chosen_model = min(candidates, key=lambda m: model_costs[m])
    return chosen_model
   
def route(user_text, user_tier):
    query_type, domain, user_tier, token_count = text_input(user_text, user_tier)
    selected_model = select_model(query_type, user_tier, token_count)
    return selected_model, domain


if __name__ == '__main__':
    #print(route('how to start learning AI for non tech users?','enterprise'))
    # print(route('code_generation', 'premium', count_tokens("短问题")))
    # print(route('analysis', 'free', count_tokens("x"*300000)))
    print(classifyChat('how to start learning AI for non tech users?'))
    print(classifyChat('Fedex says it picked up my package 10 days ago but there is no update.'))
    print(classifyChat('Does a rabbit swim at all?'))
    