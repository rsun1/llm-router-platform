from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import io
from fastapi.responses import StreamingResponse
import json
from datetime import datetime

app = FastAPI()

@app.get('/health')
def health_check():
    return {"status": "Healthy"}