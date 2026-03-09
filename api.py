from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import chromadb
import PyPDF2
import os
import io
from dotenv import load_dotenv
from tinydb import TinyDB, Query # <-- NEW: Import TinyDB

# Unlock the vault
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./pdf_db")

app = FastAPI(title="AI Financial Researcher API", version="1.0")

# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------
class FinancialMetrics(BaseModel):
    company_name: str = Field(description="The name of the company")
    fiscal_period: str = Field(description="The quarter or year of the report (e.g., Q4 2023)")
    total_revenue: str = Field(description="Total revenue reported, including the currency/scale")
    net_income: str = Field(description="Total net income")
    earnings_per_share: str = Field(description="EPS (Earnings Per Share)")
    forward_guidance: str = Field(description="A 1-sentence summary of future guidance")

# Initialize TinyDB for chat memory
db = TinyDB('nosql_chat_db.json')
chat_collection = db.table('messages')

class ChatRequest(BaseModel):
    question: str

# ---------------------------------------------------------
# ENDPOINT 1: Upload and Index PDF
# ---------------------------------------------------------
@app.post("/api/upload/{user_id}")
async def upload_document(user_id: str, file: UploadFile = File(...)):
    # Read the uploaded PDF file from memory
    pdf_content = await file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() + "\n\n"
        
    # Chunking logic
    chunk_size = 1000
    overlap = 200
    text_chunks = [raw_text[i : i + chunk_size].strip() for i in range(0, len(raw_text), chunk_size - overlap) if len(raw_text[i : i + chunk_size].strip()) > 100]

    # Database operations
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    try:
        collection.delete(where={"user_id": user_id})
    except Exception:
        pass 
        
    embeddings_list = []
    ids_list = []
    metadatas_list = []
    
    for i, chunk in enumerate(text_chunks):
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=chunk
        )
        embeddings_list.append(response.embeddings[0].values)
        ids_list.append(f"{user_id}_chunk_{i}")
        metadatas_list.append({"user_id": user_id, "source_file": file.filename})
    
    collection.add(documents=text_chunks, embeddings=embeddings_list, metadatas=metadatas_list, ids=ids_list)
    
    return {"message": f"Successfully indexed {len(text_chunks)} chunks for {user_id}", "filename": file.filename}

# ---------------------------------------------------------
# ENDPOINT 2: Extract Structured JSON
# ---------------------------------------------------------
@app.get("/api/extract/{user_id}", response_model=FinancialMetrics)
def extract_metrics(user_id: str):
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    query_embed = client.models.embed_content(
        model='gemini-embedding-001',
        contents="financial summary total revenue net income EPS forward guidance"
    ).embeddings[0].values
    
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=5,
        where={"user_id": user_id}
    )
    
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No documents found for this user in the vault.")
        
    context = "\n\n".join(results['documents'][0])
    
    json_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FinancialMetrics,
        temperature=0.0
    )
    
    prompt = f"Extract the core financial metrics from this context:\n\n{context}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=json_config
    )
    
    import json
    return json.loads(response.text)

# ---------------------------------------------------------
# ENDPOINT 3: Chat with Agent (RAG + Web Search)
# ---------------------------------------------------------
@app.post("/api/chat/{user_id}")
def chat_with_agent(user_id: str, request: ChatRequest):
    # 1. Save the user's new question to TinyDB
    chat_collection.insert({"user_id": user_id, "role": "user", "content": request.question})
    
    # 2. Retrieve PDF Context from ChromaDB
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    question_embedding = client.models.embed_content(
        model='gemini-embedding-001',
        contents=request.question
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3, 
        where={"user_id": user_id}
    )
    
    combined_context = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        combined_context = "\n\n---\n\n".join(results['documents'][0])

    # 3. Retrieve Chat History from TinyDB
    MessageQuery = Query()
    user_docs = chat_collection.search(MessageQuery.user_id == user_id)
    
    # Build history string (excluding the brand new question we just inserted)
    conversation_string = ""
    for doc in user_docs[:-1]: 
        speaker = "User" if doc["role"] == "user" else "Assistant"
        conversation_string += f"{speaker}: {doc['content']}\n\n"

    # 4. The Agent Prompt
    final_prompt = f"""
    You are a Senior Financial Research Analyst. You have access to a PDF CONTEXT and a Google Search tool.
    
    ### PREVIOUS CONVERSATION HISTORY:
    {conversation_string}

    ### INSTRUCTIONS:
    - Answer the user's latest question using the CONTEXT and your web search tool.
    - CRITICAL: Read the PREVIOUS CONVERSATION HISTORY to understand follow-up questions.
    - Keep answers concise and strictly factual.
    
    ### PDF CONTEXT:
    {combined_context}

    ### LATEST USER QUESTION:
    {request.question}
    """
    
    agent_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    
    # 5. Generate Answer
    final_answer = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=final_prompt,
        config=agent_config
    )
    
    # 6. Save Assistant Answer to TinyDB
    chat_collection.insert({"user_id": user_id, "role": "assistant", "content": final_answer.text})
    
    # 7. Extract Search Metadata
    search_queries = []
    try:
        queries = final_answer.candidates[0].grounding_metadata.web_search_queries
        if queries:
            search_queries = list(queries)
    except Exception:
        pass
        
    # Return the payload to the frontend
    return {
        "answer": final_answer.text,
        "search_queries": search_queries
    }

# ---------------------------------------------------------
# ENDPOINT 4: Get Chat History
# ---------------------------------------------------------
@app.get("/api/history/{user_id}")
def get_chat_history(user_id: str):
    MessageQuery = Query()
    user_docs = chat_collection.search(MessageQuery.user_id == user_id)
    # Return just the role and content for the frontend
    return [{"role": doc["role"], "content": doc["content"]} for doc in user_docs]