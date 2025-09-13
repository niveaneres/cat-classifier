from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.services.model import ModelHandler
import base64

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
model = ModelHandler()

@app.get("/health")
def health_check():
    return{"Alive": 200}

@app.get("/")
def health_check():
    return{"Alive": 200}

@app.post("/")
async def this_is_a_cat(request: Request):
    data = await request.json()
    img_base64 = data.get("image")

    if not img_base64:
        return JSONResponse(content={"error": "Image not found"}, status_code=400)

    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]
    img_bytes = base64.b64decode(img_base64)
   
    results = model.inference(img_bytes)
    return JSONResponse(content={"result": results})
