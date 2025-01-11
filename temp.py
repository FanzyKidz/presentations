import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
import uvicorn

# Allow asyncio to run in Jupyter Notebook
nest_asyncio.apply()

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

# Function to run the FastAPI app in Jupyter
def start_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Start the API
start_api()
