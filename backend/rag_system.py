import os
import random
import requests
from pathlib import Path
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext  
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.prompts import PromptTemplate
import deeplake
import numpy as np
import json

EMBED_DIM = 1536

class Float32Embedding(OpenAIEmbedding):
    def _to_f32(self, emb): 
        return np.array(emb, dtype=np.float32).tolist()
    
    def get_text_embedding(self, text): 
        return self._to_f32(super().get_text_embedding(text))
    
    def get_query_embedding(self, q):   
        return self._to_f32(super().get_query_embedding(q))
    
    def get_text_embedding_batch(self, texts):  
        return [self._to_f32(e) for e in super().get_text_embedding_batch(texts)]
    
    def get_query_embedding_batch(self, qs):    
        return [self._to_f32(e) for e in super().get_query_embedding_batch(qs)]

def normalize_metadata(meta):
    if meta is None: 
        return {}
    if isinstance(meta, dict):
        return {k: (v if isinstance(v, (str,int,float)) or v is None else json.dumps(v))
                for k,v in meta.items()}
    return {"value": str(meta)}

def download_paul_graham_data():
    """
    Download Paul Graham essay data - same as your existing code
    """
    # Create paul_graham directory if it doesn't exist
    paul_graham_dir = Path("./paul_graham")
    paul_graham_dir.mkdir(exist_ok=True)
    
    essay_path = paul_graham_dir / "paul_graham_essay.txt"
    
    # Download the essay if it doesn't exist
    if not essay_path.exists():
        print("📥 Downloading Paul Graham essay...")
        url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            with open(essay_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✅ Successfully downloaded essay to {essay_path}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading essay: {e}")
            raise
    else:
        print(f"📚 Essay already exists at {essay_path}")
    
    return paul_graham_dir

def setup_rag_system():
    """
    Setup RAG system with automatic data download
    """
    # Download Paul Graham data (your existing approach)
    paul_graham_dir = download_paul_graham_data()
    
    # Load documents (your existing code)
    print("📖 Loading documents...")
    documents = SimpleDirectoryReader(str(paul_graham_dir)).load_data()
    print(f"✅ Loaded {len(documents)} documents")
    
    # Initialize service context
    service_context = ServiceContext.from_defaults(chunk_size=512, chunk_overlap=64)
    node_parser = service_context.node_parser
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"📝 Created {len(nodes)} text chunks")
    
    embed_model = Float32Embedding()
    
    # Setup Deep Lake vector store
    dataset_path = "hub://lemojames101/LlamaIndex_paulgraham_essays"
    
    try:
        # Try to load existing dataset first
        vector_store = DeepLakeVectorStore(dataset_path=dataset_path)
        print("📚 Loaded existing dataset from DeepLake")
    except:
        # Create new dataset if doesn't exist
        print("🏗️ Creating new DeepLake dataset...")
        try:
            deeplake.delete(dataset_path)
        except:
            pass
            
        vector_store = DeepLakeVectorStore(
            dataset_path=dataset_path, 
            overwrite=True,
            tensor_params=[
                {"name": "embedding", "dtype": "float32", "htype": "embedding"},
                {"name": "metadata",  "htype": "json"},
                {"name": "text",      "dtype": "str"},
            ],
        )
        
        # Monkey-patch add to sanitize nodes
        old_add = vector_store.add
        def safe_add(nodes, **kwargs):
            for n in nodes:
                n.embedding = np.array(getattr(n, "embedding", np.zeros(EMBED_DIM)), dtype=np.float32)
                if hasattr(n, "metadata"): 
                    n.metadata = normalize_metadata(n.metadata)
                if hasattr(n, "text") and n.text is not None: 
                    n.text = str(n.text)
            return old_add(nodes, **kwargs)
        vector_store.add = safe_add
        
        # Build index
        print("🔧 Building vector index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True)
        print("✅ Built new vector index")
    
    # Create index from existing store
    if 'index' not in locals():
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context, embed_model=embed_model)
    
    # Create query engine with custom prompt to stay within scope
    custom_prompt = PromptTemplate(
        "You are an AI assistant that answers questions exclusively based on Paul Graham's essays. "
        "Only provide information that can be found in the provided context from Paul Graham's writings. "
        "If the question cannot be answered using the provided context, respond with 'I don't have information about this topic in Paul Graham's essays.' "
        "Do not use general knowledge or information from other sources.\n\n"
        "Context information is below:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer based only on the context above: "
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        text_qa_template=custom_prompt,
        streaming=True
    )
    
    print("🎯 RAG system setup complete!")
    return query_engine

def is_relevant_response(response_text, question):
    """
    Check if the response is relevant to Paul Graham's essays
    Returns True if relevant, False if off-topic
    """
    # Keywords that indicate the response is from Paul Graham's content
    pg_indicators = [
        "paul graham", "y combinator", "yc", "startup", "lisp", "viaweb",
        "hacker", "programming", "essay", "venture capital", "silicon valley",
        "founder", "entrepreneur", "technology", "software", "computer science",
        "arc", "painting", "art", "harvard", "mit", "writer", "investor"
    ]
    
    # Generic responses that indicate lack of relevant information
    irrelevant_indicators = [
        "i don't know", "i'm not sure", "i don't have information",
        "i cannot find", "no information", "not mentioned", "unclear",
        "i apologize", "i'm sorry", "i don't have access",
        "based on general knowledge", "in general", "typically",
        "i don't have information about this topic in paul graham's essays"
    ]
    
    response_lower = response_text.lower()
    question_lower = question.lower()
    
    # Check if response contains irrelevant indicators
    if any(indicator in response_lower for indicator in irrelevant_indicators):
        return False
    
    # Check if response or question contains Paul Graham related content
    if any(indicator in response_lower or indicator in question_lower for indicator in pg_indicators):
        return True
    
    # Additional check: if response is very short and generic, it's likely irrelevant
    if len(response_text.strip()) < 50:
        return False
    
    # If response mentions specific concepts that could be in essays, consider it relevant
    concept_indicators = [
        "according to", "mentioned", "discussed", "explained", "wrote about",
        "believes", "argues", "suggests", "recommends", "experience", "opinion"
    ]
    
    if any(indicator in response_lower for indicator in concept_indicators):
        return True
    
    # Default to irrelevant if no clear indicators
    return False

def get_scope_limitation_message():
    """
    Return a polite message explaining the system's scope
    """
    messages = [
        "I'm sorry, but I can only provide information based on Paul Graham's essays. Please ask me about topics related to startups, programming, Y Combinator, or other subjects Paul Graham has written about.",
        
        "I specialize in answering questions about Paul Graham's essays and insights. Could you please ask something related to startups, programming, entrepreneurship, or other topics he's covered in his writings?",
        
        "My knowledge is limited to Paul Graham's essays and writings. I'd be happy to help with questions about startups, Y Combinator, programming languages, or other topics he's discussed.",
        
        "I can only answer questions based on Paul Graham's essays. Please feel free to ask about startups, programming, venture capital, or any other topics from his writings.",
        
        "I'm designed to answer questions specifically about Paul Graham's essays and insights. Could you ask something related to entrepreneurship, programming, or other subjects he's written about?"
    ]
    
    return random.choice(messages)

def query_rag_system(query_engine, question):
    # First, get the RAG response
    streaming_response = query_engine.query(question)
    
    # Convert streaming response to string
    response_text = ""
    try:
        for chunk in streaming_response.response_gen:
            response_text += chunk
    except:
        # Fallback for non-streaming response
        response_text = str(streaming_response)
    
    # Check if the response is relevant to Paul Graham's content
    if is_relevant_response(response_text, question):
        return response_text
    else:
        return get_scope_limitation_message()