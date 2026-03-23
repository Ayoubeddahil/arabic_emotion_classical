from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from .preprocess import clean_arabic_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
)


model = None
vectorizer = None
model_path = 'app/models/best_model.pkl'
vec_path = 'app/models/vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vec_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    print("Modèle chargé")

class TexteRequest(BaseModel):
    texte: str

@app.post("/predict")
def predict(request: TexteRequest):
    if model is None:
        return {"erreur": "Modèle non chargé"}
    
    
    cleaned = clean_arabic_text(request.texte)
    vec = vectorizer.transform([cleaned])
    
    
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0].tolist() if hasattr(model, "predict_proba") else []
    
    return {
        "texte_original": request.texte,
        "texte_nettoye": cleaned,
        "prediction": str(prediction),
        "probabilites": probabilities,
        "classe": str(prediction)
    }

@app.get("/")
def root():
    return {"message": "API NLP Arabe", "modele_charge": model is not None}

@app.get("/modeles")
def get_modeles():
    return {"meilleur_modele": str(type(model).__name__) if model else "inconnu"}