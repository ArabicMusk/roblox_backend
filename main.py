import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# We look for the API key in the server settings
HF_TOKEN = os.getenv("HF_TOKEN")
# Using the high-quality BART model hosted on Hugging Face's free servers
API_URL = "https://api-inference.huggingface.co/pipeline/zero-shot-classification/facebook/bart-large-mnli"

GAME_SECTORS = ["Technology", "Retail & Shopping", "Food & Hospitality", "Service & Labor"]

class IdeaRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/classify")
def classify_idea(request: IdeaRequest):
    if not HF_TOKEN:
        return {"status": "error", "message": "Missing HF_TOKEN environment variable on Render."}

    payload = {
        "inputs": request.text,
        "parameters": {"candidate_labels": GAME_SECTORS}
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        output = response.json()
        
        # Format the data cleanly for Roblox
        return {
            "status": "success",
            "category": output["labels"][0],
            "confidence": round(output["scores"][0] * 100, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
