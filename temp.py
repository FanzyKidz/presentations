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











from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

app = FastAPI()

class PromptRequest(BaseModel):
    query: str
    max_tokens: int = 1000

# Initialize local LLM
model, tokenizer = load("./Users/u801658/unilang/Lama")
prompt_cache = make_prompt_cache(model)

# Initialize ChromaDB with embeddings from the same local LLM
embeddings = HuggingFaceEmbeddings(model_name="./Users/u801658/unilang/Lama")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=model,  # Use the local LLM
    retriever=vectorstore.as_retriever()
)

@app.post("/generate")
def generate_content(request: PromptRequest):
    try:
        # Generate response using RAG model
        rag_response = qa_chain.run(request.query)
        
        # Generate response using local LLM
        local_llm_response = generate(
            model=model,
            tokenizer=tokenizer,
            max_tokens=request.max_tokens,
            prompt=request.query,
            verbose=True
        )

        # Combine outputs
        combined_response = {
            "rag_response": rag_response,
            "local_llm_response": local_llm_response
        }

        return JSONResponse(content=combined_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this FastAPI application:
# 1. Save this script to a file, e.g., `main.py`
# 2. Install necessary dependencies: `pip install fastapi uvicorn langchain chromadb mlx_lm`
# 3. Ensure ChromaDB is properly initialized and populated with data.
# 4. Run the application: `uvicorn main:app --reload`
# 5. Access the API documentation at `http://127.0.0.1:8000/docs`
