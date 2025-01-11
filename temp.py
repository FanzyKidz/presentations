from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mlx_lm

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

# Initialize your local LLM using mlx_lm
model = mlx_lm.load_model("path_to_your_local_model")

def stream_response(prompt: str, max_tokens: int, temperature: float):
    try:
        for chunk in model.stream_generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature):
            yield chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-stream")
def generate_stream(request: PromptRequest):
    return StreamingResponse(stream_response(request.prompt, request.max_tokens, request.temperature), media_type="text/plain")

# To run this FastAPI application:
# 1. Save this script to a file, e.g., `main.py`
# 2. Install FastAPI and Uvicorn using pip: `pip install fastapi uvicorn mlx_lm`
# 3. Run the application using: `uvicorn main:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs`
