from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import mlx.core as mx
from mlx_lm import load, generate
import json

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
    previous_text = ""
    
    try:
        for token in generate(model, tokenizer, prompt=prompt, max_tokens=500):
            tokens.append(token)
            # Convert token to text and yield only the new content
            current_text = tokenizer.decode(mx.array(tokens))
            new_text = current_text[len(previous_text):]
            previous_text = current_text
            
            if new_text:
                yield f"data: {json.dumps({'text': new_text})}\n\n"
        
        # Send end of stream
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error in stream generation: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    return StreamingResponse(
        process_stream(message, files),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )
