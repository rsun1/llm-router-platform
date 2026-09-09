from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import io
from fastapi.responses import StreamingResponse
import json
from datetime import datetime
from pydantic import BaseModel, Field 
from llm_router_part7_langgraph import graph 
import time 

class RouteRequest(BaseModel):
    query_text: str = Field(min_length=1)
    user_tier: str 
    user_id:str 

app = FastAPI()

@app.get('/health')
def health_check():
    return {"status": "Healthy"}




@app.post('/route')
def route(request:RouteRequest):
    res = graph.invoke({
        'query_text': request.query_text,
        'user_tier':request.user_tier,
        'user_id':request.user_id,
        'start_time': time.time()
    })
    return {
        'response': res.get('response'),
        'selected_model': res.get('selected_model'),
        'fallback_model': res.get('fallback_model'),
        'adapter_id': res.get('adapter_id'),
        'latency_ms': res.get('latency_ms'),
        'status': res.get('status'),
    }
    