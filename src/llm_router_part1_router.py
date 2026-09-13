import os 
import yaml 
from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage
from classify_lora import _classify_lora
from router_prompt import DOMAIN_MAP


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

# TYPE_MODELS = {
#     'code_generation': ['gpt-4-turbo', 'mistral-7b'],
#     'analysis': ['claude-3.5-sonnet','llama-3.1-70b'],
# }
CLASSIFIER = load_config()['router'].get('classifier','lora')
def count_tokens(text):
    n_tokens = len(text)//4 
    return n_tokens

llm = ChatOllama(
    model='llama3.1',
    temperature=0,
    num_predict=5
)
def _classify_int(user_text):
    prompt = """
        Determine the type of user question: Only int values ​​can be output: 0general / 1code / 2legal / 3customer_service
        Respond with exactly one character: 0, 1,2 or 3. No explanation, no punctuation. 
        Code (1) means it is asking for assistance with computer coding languages, project architecture, dependencies, dealing with errors
        , databases, or structures.
        Legal (2) means the it is asking about details in legal or patent, or legislation, etc, such as whether something
        is legal or legally effective, or about rights and previliges, etc.
        Customer service (3) means the it is asking for help on orders, or regarding products, or charges and refunds, etc.
        Other types are general (0), such as knowledge extraction about history, arts, life, common sense etc.
        Examples:
        how to start learning AI for non tech users? label: '0'
        Give me a barebone structure of a two tower model. label: '1'
        How should my team set up a patent review process? label: '2'
        Fedex says it has picked up my package but I don't see more tracking update. Where is my order? label: '3'
        question:{query}
        label:
    """.format(query=user_text)
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # pos = res.content.find("Domain")
    # first_part_query_type = res.content[:pos].strip().lower().strip('".') 
    # second_part_domain_type = res.content[pos:].strip().lower().strip('".')     
    
    # classified = ['code_generation','analysis','general','summarization']
    # domain = ['code','customer_service','general']
    # result=['general','general']
    
    # for c in classified:
    #     if c in first_part_query_type:
    #         result[0] = c

    # for d in domain:
    #     if d in second_part_domain_type:
    #         result[1] = d 
    # return result
    for ch in res:
        if ch in '0123': 
            return int(ch)
    return 0
def classifyChat(user_text):
    if CLASSIFIER == 'ollama' :
        choose_model = _classify_int
    elif CLASSIFIER == 'lora':
        choose_model = _classify_lora
    classify_int = choose_model(user_text)
    return DOMAIN_MAP[classify_int]


def text_input(user_text,user_tier):
    domain = classifyChat(user_text)
    token_count = count_tokens(user_text)
    return domain, user_tier, token_count

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
    # print(classifyChat('how to start learning AI for non tech users?'))
    # print(classifyChat('How should my team set up a patent review process?'))
    # print(classifyChat('Does a rabbit swim at all?'))
    print(classifyChat('Shouls I use list or dict for my use case here in python?'))
    print(classifyChat('How long does a patent protection period last in medication development?'))
    print(classifyChat('I saw you launched a discount deal today. Can I ask for a partial refund of my order I just placed yesterday?'))
    print(classifyChat('When is the last time France went into war with Italy?'))
    
    print(classifyChat('Help me generate a python script that tracks plane ticket pricing for the following route.'))
    print(classifyChat('Who to contact to ask about the rules on the process of roling out this feature for our D2 customers?'))
    print(classifyChat("I've been reading your return policy. Is it legal that you don't allow any return on the books sold from this website?"))
    print(classifyChat('What year did Ombamacare policy come into effect?'))
    
    print(classifyChat('Would it be better if I download uv for my project? Why?'))
    print(classifyChat('When was the Lindsay Clancy case happened and when was the trial?'))
    print(classifyChat("I keep getting 404 error on placing order webpage, what's going on?"))
    print(classifyChat('Help me find the exact policy within company rulebook on D2 customer returns.'))