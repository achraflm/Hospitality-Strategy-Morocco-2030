import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ajouter le répertoire backend au path python pour résoudre les imports de src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api import forecast, roi, monte_carlo

app = FastAPI(
    title="Morocco Tourism Investment Intelligence Platform API",
    description="API Python pour le forecasting touristique et la simulation financière. "
                "Inclut désormais un environnement **Autoresearch** (inspiré par les recherches "
                "de A. Karpathy) permettant à des Agents IA d'optimiser les modèles en continu.",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dans un environnement de production, restreindre aux domaines appropriés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enregistrement des routeurs d'API
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecasting"])
app.include_router(roi.router, prefix="/api/roi", tags=["ROI Simulator"])
app.include_router(monte_carlo.router, prefix="/api/monte-carlo", tags=["Monte Carlo"])


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Morocco Tourism Investment Intelligence Platform API represents ready.",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
