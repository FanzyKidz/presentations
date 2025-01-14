from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import mlx.core as mx
from mlx_lm import load, generate
import os

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
model, tokenizer = load("mlx-community/Mistral-7B-v0.1-hf-4bit-mlx")

async def process_stream(message: str, files: List[UploadFile]):
    # Process files if present
    file_contents = []
    for file in files:
        content = await file.read()
        file_contents.append(f"File '{file.filename}' content: {content.decode('utf-8', errors='ignore')[:1000]}...")

    # Combine message and file contents
    system_prompt = "You are a helpful AI assistant."
    user_content = message
    if file_contents:
        user_content += "\n\nAttached files:\n" + "\n".join(file_contents)

    # Prepare prompt
    prompt = f"{system_prompt}\n\nUser: {user_content}\n\nAssistant:"

    # Generate response with streaming
    tokens = []
    for token in generate(model, tokenizer, prompt=prompt, max_tokens=500):
        tokens.append(token)
        # Convert token to text and yield
        text = tokenizer.decode(mx.array(tokens))
        yield text

@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    return StreamingResponse(
        process_stream(message, files),
        media_type='text/event-stream'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
