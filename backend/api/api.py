from fastapi import APIRouter, File, UploadFile
import os


router = APIRouter()

@router.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    from main import pipeline
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        prediction = pipeline.classify_song(file_location)
    finally:
        os.remove(file_location)

    return prediction
