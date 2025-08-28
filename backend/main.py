from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv
from rag_system import setup_rag_system, query_rag_system

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://sunny-tulumba-bb8a69.netlify.app",  # Netlify frontend
],  # React dev server
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# Global state
query_engine = None
rag_ready = False

async def initialize_rag():
    global query_engine, rag_ready
    try:
        print("🚀 Setting up RAG system in background...")
        query_engine = setup_rag_system()
        rag_ready = True
        print("✅ RAG system ready!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")

@app.on_event("startup")
async def startup_event():
    # Spawn async task so startup is non-blocking
    asyncio.create_task(initialize_rag())

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    if not rag_ready or query_engine is None:
        raise HTTPException(status_code=503, detail="RAG system still initializing, try again later.")
    
    try:
        response = query_rag_system(query_engine, request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": rag_ready}
