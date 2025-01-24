from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api import router as api_router
import pandas as pd
from scripts.config import DATAFRAME_PATH
from predict.pipeline import Pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

pipeline = None

@app.on_event("startup")
async def initialize_pipeline():
    global pipeline
    df_music = pd.read_csv(DATAFRAME_PATH)
    pipeline = Pipeline(df_music)
    print("Pipeline initialized and models trained!")

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Music Genre Classifier API"}