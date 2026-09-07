import os 
import json 
import yaml
from datetime import datetime
import hashlib 

rootpath = os.path.join(os.path.dirname(os.path.dirname(__file__)))

def load_config():
    config = os.path.join(rootpath,'config', 'config.yaml')
    with open(config) as f:
        return yaml.safe_load(f)

def load_registry():
    config = load_config()
    registry_file = os.path.join(rootpath, config['adapters']['registry_path'], 'registry.json')
    
    if not os.path.exists(registry_file):
        return {}
    
    with open(registry_file) as f:
        return json.load(f)

def save_registry(registry):
    config = load_config()
    registry_file = os.path.join(rootpath, config['adapters']['registry_path'], 'registry.json')
    
    os.makedirs(os.path.dirname(registry_file), exist_ok=True)
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)

def register_adapter(base_model, domain, version, canary_pct=0.1):
    adapter_id = f'{base_model}-{domain}-{version}'
    registry = load_registry()
    registry[adapter_id] = {
            "base_model": base_model,
            "domain": domain,
            "version": version,
            "status": "canary",
            "canary_pct": canary_pct,
            "created_at": datetime.now().isoformat(),
            "updated_at": None, 
    }
    save_registry(registry)
    return adapter_id

def in_canary(user_id, canary_pct):
    h = hashlib.md5(user_id.encode()).hexdigest()
    bucket = int(h, 16) % 100
    return bucket < canary_pct *100


def select_adapter(base_model, domain, user_id):
    registry = load_registry()
    for adapter_id, info in registry.items():
        if  info['base_model'] != base_model or  info["domain"] != domain:
            continue 
        if info["status"] == "production":
            return adapter_id
        if info["status"] == "canary" and in_canary(user_id,info["canary_pct"]):
            return adapter_id
            
def promote_adapter(adapter_id):
    registry = load_registry()
    if adapter_id not in registry:
        raise Exception("adapter_id does not exist in registry. Please add it in first before trying again.")
    registry[adapter_id]["status"] = "production"
    registry[adapter_id]["updated_at"] = datetime.now().isoformat()
    save_registry(registry)

def rollback_adapter(adapter_id):
    registry = load_registry()
    if adapter_id not in registry:
        raise Exception("adapter_id does not exist in registry. Please check again.")
    registry[adapter_id]["status"] = "rolled_back"
    registry[adapter_id]["updated_at"] = datetime.now().isoformat()
    save_registry(registry)




if __name__ == "__main__":
    register_adapter('mistral-7b','customer_service','v1')
    register_adapter('mistral-7b','general','v1',0.1)
    promote_adapter('mistral-7b-general-v1')