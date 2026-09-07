from llm_router_part1_router import route
from llm_router_part2_inference import call_model

def run_pipeline(user_text, user_tier):
    selected_model, _ = route(user_text, user_tier)
    response = call_model(selected_model, user_text, user_tier)
    return selected_model, response 

if __name__ == '__main__':
    print(run_pipeline('give me a python code for random forest','free'))