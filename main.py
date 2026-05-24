from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Upgrade to a zero-shot classifier (Still fits on free tiers, but can categorize anything)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the business categories you want your Roblox game to use
GAME_SECTORS = ["Technology", "Retail & Shopping", "Food & Hospitality", "Service & Labor"]

class IdeaRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_idea(request: IdeaRequest):
    # The AI evaluates the text against your specific game sectors
    model_output = classifier(request.text, candidate_labels=GAME_SECTORS)
    
    # Extract the top matched category and the confidence score
    top_label = model_output["labels"][0]
    confidence_score = model_output["scores"][0]
    
    return {
        "status": "success",
        "category": top_label,
        "confidence": round(confidence_score * 100, 2) # e.g., 94.5%
    }

