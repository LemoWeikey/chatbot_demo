from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_system import setup_rag_system, query_rag_system

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# Initialize RAG system on startup
query_engine = None

@app.on_event("startup")
async def startup_event():
    global query_engine
    print("🚀 Setting up RAG system...")
    query_engine = setup_rag_system()
    print("✅ RAG system ready!")

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    try:
        if query_engine is None:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        response = query_rag_system(query_engine, request.question)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": query_engine is not None}