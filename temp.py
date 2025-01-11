from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 10000

# Initialize your local LLM using mlx_lm
model, tokenizer = load("./Users/u801658/unilang/Lama")
prompt_cache = make_prompt_cache(model)

@app.post("/generate-content")
def generate_content(request: PromptRequest):
    try:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            max_tokens=request.max_tokens,
            prompt=request.prompt,
            verbose=True
        )
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this FastAPI application:
# 1. Save this script to a file, e.g., `main.py`
# 2. Install FastAPI and Uvicorn using pip: `pip install fastapi uvicorn mlx_lm`
# 3. Run the application using: `uvicorn main:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs`
