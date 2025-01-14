from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import mlx.core as mx
from mlx_lm import load, generate
import json
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (Mistral 7B by default)
try:
    model, tokenizer = load("mlx-community/Mistral-7B-v0.1-hf-4bit-mlx")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

async def process_request(message: str, files: Optional[List[UploadFile]] = None):
    try:
        # Process files if present
        file_contents = []
        if files:
            for file in files:
                content = await file.read()
                file_contents.append(f"File '{file.filename}' content: {content.decode('utf-8', errors='ignore')[:1000]}...")

        # Prepare prompt
        prompt = f"User: {message}"
        if file_contents:
            prompt += "\n\nAttached files:\n" + "\n".join(file_contents)
        prompt += "\n\nAssistant:"

        # Generate response
        generated_text = generate(model, tokenizer, prompt=prompt, max_tokens=500)

        if isinstance(generated_text, str):
            return {"text": generated_text}
        else:
            raise ValueError(f"Unexpected output from generate: {type(generated_text)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: List[UploadFile] = File(None)
):
    print(f"Received message: {message}")  # Debug log
    print(f"Received files: {files}")  # Debug log

    if not message or message.strip() == "":
        raise HTTPException(status_code=400, detail="Message is required")

    response = await process_request(message, files)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
