import time
import os
import csv 
from datetime import datetime, timedelta
import yaml
import pandas as pd 
from llm_router_part1_router import route
from llm_router_part2_inference import call_model
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
    
    
def check_quota(user_id, user_tier):
    config = load_config()
    quotas = config['policies']['quota_enforcement']['tier_quotas'][user_tier]

    if not os.path.exists(LOG_PATH):
        return True 
    
    logs = load_data()
    user_logs = logs[logs['user_id']==user_id]

    now = datetime.now()
    daily_count = (user_logs['timestamp'] >= now - timedelta(days=1)).sum()
    hourly_count = (user_logs['timestamp'] >= now - timedelta(hours=1)).sum()
    
    if daily_count >= quotas['daily']:
        return False
    if hourly_count >= quotas['hourly']:
        return False
    return True
    
def check_circuit_breaker(selected_model):
    config = load_config()
    failure_max = config['policies']['circuit_breaker']['failure_threshold']
    open_criteria = config['policies']['circuit_breaker']['half_open_after']
    
    if not os.path.exists(LOG_PATH):
        return True
    
    data_history = load_data()
    model_history=data_history[data_history['selected_model']==selected_model].sort_values('timestamp', ascending=False)

    failure_count = 0 
    last_failure_time = None
    
    for _, row in model_history.iterrows():
        if row['status'] == 'error' or pd.notna(row['fallback_model']):
            failure_count += 1
            if last_failure_time is None:
                last_failure_time = row['timestamp']
        else:
            break 
    if failure_count < failure_max:
        return True 
    
    seconds_since_failure = (datetime.now() - last_failure_time).total_seconds()
    if seconds_since_failure >= open_criteria:
        return True 
    return False
    
def monitored_call(user_text, user_tier, user_id):
    start_time = time.time() 
    timestamp = datetime.now().isoformat()
    selected_model, domain = route(user_text, user_tier)
    
    if not check_quota(user_id, user_tier):
        status = 'blocked'
        error_message = 'Exceeded quota'
        response = None
        adapter_id = None
    elif not check_circuit_breaker(selected_model):
        status = 'blocked'
        error_message = 'Triggered circuit breaker'
        response = None
        adapter_id = None
    else:
        try:
            adapter_id = select_adapter(selected_model,domain, user_id)
            response = call_model(selected_model, user_text, user_tier)
            status = 'success'
            error_message = None
        except Exception as e:
            status = 'error'
            error_message = str(e)
            response = None
    end_time = time.time() 
    
    latency = (end_time - start_time)*1000
    config = load_config()
    sla_limit = config["policies"]["latency_slas"][user_tier]
    sla_met = latency <= sla_limit
    
    print(f'Latency: {latency}ms')
    session = {
        'user_id': user_id,
        'query_text': user_text,
        'user_tier': user_tier,
        'selected_model': selected_model,
        'classified_domain': domain,
        'adapter_id': adapter_id,
        'latency_ms': latency,
        'status': status,
        'error_message': error_message,
        'timestamp': timestamp,
        'sla_met': sla_met
    }
    print(f'Session info: {session}')
    return response, session

def save_session(session):
    file_exists = os.path.exists(LOG_PATH)
    
    if file_exists:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            existing_header = f.readline().strip().split(',')
        if existing_header != list(session.keys()):
            raise Exception(
                "'call_logs.csv' existing header does not match session due to changed schema."
                "delete the current call_logs.csv before rerun."
            )
    
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=session.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(session)

    
if __name__ == '__main__':
    response, session = monitored_call('give me a code of sample transformer','enterprise','user_201')
    save_session(session)
    check_quota('user_201','enterprise')
    response, session = monitored_call('a python code sample for running an agent eval which already has 50 sets of golden questions.','free','user_202')
    save_session(session)
    response, session = monitored_call('hello','free','user_202')
    save_session(session)
    response, session = monitored_call('is Sydney a capital?','premium','user_203')
    save_session(session)