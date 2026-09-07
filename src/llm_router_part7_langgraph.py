from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal 
from llm_router_part1_router import text_input, select_model, pick_fallback_model
from llm_router_part2_inference import call_model, TOKEN_MODEL_MAP, check_cost
from utils.token_tools import count_messages_tokens
import os 
import pandas as pd
import yaml
from llm_router_part4_monitor import save_session, check_quota, check_circuit_breaker
import time
from datetime import datetime, timedelta
from llm_router_part6_adapters import select_adapter

configpath = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config')
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data','call_logs.csv')

def load_config():
    config = os.path.join(configpath, 'config.yaml')
    with open(config) as f:
        return yaml.safe_load(f)

def load_data():
    logs = pd.read_csv(LOG_PATH)
    logs['timestamp'] = pd.to_datetime(logs['timestamp'])
    return logs


class RouteState(TypedDict):
    user_id: str 
    query_text: str 
    user_tier: str 
    query_type: str
    token_count: int
    selected_model: str 
    classified_domain: str
    response: str
    adapter_id: str 
    latency_ms: float 
    status: str 
    error_message: str 
    timestamp:  str
    sla_met: bool
    start_time: float
    check_passed: bool
    fallback_model: str

def classify(state: RouteState):
    user_text = state['query_text']
    user_tier = state['user_tier']
    query_type, domain, _, token_count = text_input(user_text, user_tier)
    return {
       'query_type': query_type,
       'classified_domain': domain,
       'token_count': token_count
    }

def graph_select_model(state: RouteState):
    query_type = state['query_type']
    user_tier = state['user_tier']
    token_count = state['token_count']
    selected_model = select_model(query_type, user_tier,token_count)
    return {
        'selected_model': selected_model
    }

def graph_call_model(state: RouteState):
    selected_model = state['selected_model']
    user_text = state['query_text']
    user_tier = state['user_tier']
    try:
        response = call_model(selected_model,user_text,user_tier)
        status = 'success'
        error_message = None
    except Exception as e: 
        status = 'error'
        error_message = str(e)
        response = None
    return {
        'response': response,
        'status': status ,
        'error_message': error_message
    }

def graph_record(state: RouteState):
    latency_ms = (time.time() - state['start_time'])*1000
    timestamp = datetime.now().isoformat()
    config = load_config()
    user_tier = state['user_tier']
    sla_limit = config["policies"]["latency_slas"][user_tier]
    sla_met = latency_ms <= sla_limit
    session = {
            'user_id': state['user_id'],
            'query_text': state['query_text'],
            'user_tier': user_tier,
            'selected_model': state.get('selected_model'),
            'fallback_model': state.get('fallback_model'),
            'classified_domain': state.get('classified_domain'),
            'adapter_id': state.get('adapter_id'),
            'latency_ms': latency_ms,
            'status': state['status'],
            'error_message': state['error_message'],
            'timestamp': timestamp,
            'sla_met': sla_met
        }
    save_session(session)
    return {
        'latency_ms': latency_ms,
        'timestamp': timestamp,
        'sla_met': sla_met
    }
    
def graph_check_quota(state: RouteState):
    check_passed= check_quota(state['user_id'], state['user_tier'])
    if not check_passed:
        error_message = 'Exceeded quota'
    else:
        error_message = None
    return {
        'check_passed': check_passed,
        'error_message': error_message
    }

def graph_select_adapter(state: RouteState):
    adapter_id = select_adapter(base_model=state['selected_model'],domain=state['classified_domain'],user_id=state['user_id'])
    return {
        'adapter_id': adapter_id
    }

def reject(state: RouteState):
    status = 'blocked'
    response = None
    return {
        'status':status,
        'response': response
    }

def graph_check_breaker(state: RouteState):
    check_passed = check_circuit_breaker(state['selected_model'])
    if not check_passed:
        error_message = 'Circuit Failure. Try later.'
    else:
        error_message = None 
    return {
        'check_passed': check_passed,
        'error_message': error_message
    } 
        
def graph_check_cost(state: RouteState):
    selected_model = state['selected_model']
    token_model_name = TOKEN_MODEL_MAP.get(selected_model, selected_model)
    input_tokens = count_messages_tokens([{"role":"user","content":state['query_text']}], token_model_name)
    check_passed = check_cost(selected_model, state['user_tier'], input_tokens,token_model_name)
    if not check_passed:
        error_message = "Costs exceeding budget."
    else:
        error_message = None 
    return {
        'check_passed': check_passed,
        'error_message': error_message
    }
    

def graph_fallback(state: RouteState):
    fallback_model = pick_fallback_model(state['selected_model'],state['user_tier'])
    if not fallback_model:
        return {}
    else:    
        try:
            response = call_model(fallback_model,state['query_text'],state['user_tier'])
            status = 'success'
            error_message = f"primary model {state['selected_model']} error: {state['error_message']}"
        except Exception as e: 
            status = 'error'
            error_message = f"primary model {state['selected_model']} error: {state['error_message']} and fallback error: {str(e)}"
            response = None
        return {
            'response': response,
            'status': status ,
            'error_message': error_message,
            'fallback_model': fallback_model
        }

def router_fn(state: RouteState):
    if not state['check_passed']:
        return 'rejectNode'
    else:
        return 'modelNode'

def router_fallback(state: RouteState):
    if state['status'] == 'error':
        return 'fallbackNode'
    else:
        return 'recordNode'

workflow = StateGraph(RouteState)
workflow.add_node('classify', classify)
workflow.add_node('select_model', graph_select_model)
workflow.add_node('call_model', graph_call_model)
workflow.add_node('record', graph_record)
workflow.add_node('check_quota',graph_check_quota)
workflow.add_node('reject',reject)
workflow.add_node('check_breaker',graph_check_breaker)
workflow.add_node('check_cost', graph_check_cost)
workflow.add_node('select_adapter', graph_select_adapter)
workflow.add_node('fallback', graph_fallback)
workflow.set_entry_point('check_quota')

workflow.add_conditional_edges(
    'check_quota',
    router_fn, {
        'rejectNode': 'reject',
        'modelNode': 'classify'
    }
)
workflow.add_conditional_edges(
    'check_breaker',
    router_fn, {
        'rejectNode': 'reject',
        'modelNode': 'check_cost'
    }
)
workflow.add_conditional_edges(
    'check_cost',
    router_fn, {
        'rejectNode': 'reject',
        'modelNode': 'select_adapter'
    }
)

workflow.add_conditional_edges(
    'call_model',
    router_fallback, {
        'fallbackNode': 'fallback',
        'recordNode': 'record'
    }
)

workflow.add_edge('reject','record')
workflow.add_edge('classify', 'select_model')
workflow.add_edge('select_model','check_breaker')
workflow.add_edge('select_adapter','call_model')
workflow.add_edge('fallback','record')
workflow.add_edge('record',END)

graph = workflow.compile()

if __name__ == '__main__':
    # res = graph.invoke({
    #     'query_text': 'When is diff-in-diff analysis suitable to use?',
    #     'user_tier':'enterprise',
    #     'user_id': 'user_201',
    #     'start_time': time.time()
    # })
    # print('=========')
    # print(res['response'])  
    # res = graph.invoke({
    #     'query_text': 'When is winter in Australia?',
    #     'user_tier':'free',
    #     'user_id': 'user_202',
    #     'start_time': time.time()
    # })
    # print('=========')
    # print(res['response'])  
    res = graph.invoke({
        'query_text': 'Indoor cycling v.s. outdoor running, which is a better exercise for cardio?',
        'user_tier':'free',
        'user_id': 'user_202',
        'start_time': time.time()
    })
    print('=========')
    print(res['response'])  
    print(graph.get_graph().draw_mermaid())