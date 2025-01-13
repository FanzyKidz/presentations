from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import openai
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # Create streaming response using OpenAI
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        stream=True
    )

    # Stream the response
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

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