from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

async def process_stream(message: str, files: Optional[List[UploadFile]] = None):
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
        tokens = []
        previous_text = ""

        for token in generate(model, tokenizer, prompt=prompt, max_tokens=500):
            if isinstance(token, int):  # Ensure token is an integer
                tokens.append(token)
                current_text = tokenizer.decode(mx.array(tokens))
                new_text = current_text[len(previous_text):]
                previous_text = current_text

                if new_text:
                    yield f"data: {json.dumps({'text': new_text})}\n\n"
            else:
                print(f"Invalid token type: {token} (expected int)")

        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error in stream generation: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: List[UploadFile] = File(None)
):
    print(f"Received message: {message}")  # Debug log
    print(f"Received files: {files}")  # Debug log

    if not message or message.strip() == "":
        raise HTTPException(status_code=400, detail="Message is required")

    return StreamingResponse(
        process_stream(message, files),
        media_type='text/event-stream'
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
