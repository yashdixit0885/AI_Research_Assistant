from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import chromadb
import PyPDF2
import os
import io
from dotenv import load_dotenv
from tinydb import TinyDB, Query
from typing import List


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
    source_file: str = Field(description="The exact name of the PDF file this data came from")
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
# ENDPOINT: Get User Documents
# ---------------------------------------------------------
@app.get("/api/documents/{user_id}")
def get_user_documents(user_id: str):
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    # Fetch all metadata tags for this user
    results = collection.get(where={"user_id": user_id}, include=["metadatas"])
    
    if not results or not results["metadatas"]:
        return {"documents": []}
        
    # Use a Python Set to extract only the unique filenames
    unique_docs = set(
        meta.get("source_file") 
        for meta in results["metadatas"] 
        if meta and "source_file" in meta
    )
    
    return {"documents": list(unique_docs)}

# ---------------------------------------------------------
# ENDPOINT: Clear User Vault
# ---------------------------------------------------------
@app.delete("/api/documents/{user_id}")
def clear_user_documents(user_id: str):
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    try:
        # This acts as our "Reset" button for the user's vector space
        collection.delete(where={"user_id": user_id})
        return {"message": "Vault successfully cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vault: {str(e)}")

# ---------------------------------------------------------
# ENDPOINT: Delete Specific Document
# ---------------------------------------------------------
@app.delete("/api/documents/{user_id}/{filename}")
def delete_specific_document(user_id: str, filename: str):
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    try:
        # Delete only chunks matching BOTH the user_id AND the specific filename
        collection.delete(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"source_file": filename}
                ]
            }
        )
        return {"message": f"Successfully deleted {filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# ---------------------------------------------------------
# ENDPOINT 2: Extract Structured JSON (Multi-Document)
# ---------------------------------------------------------
@app.get("/api/extract/{user_id}", response_model=List[FinancialMetrics])
def extract_metrics(user_id: str):
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    # 1. Ask the database what files this user currently has
    results = collection.get(where={"user_id": user_id}, include=["metadatas"])
    if not results or not results["metadatas"]:
        raise HTTPException(status_code=404, detail="No documents found in the vault.")
        
    unique_files = set(meta.get("source_file") for meta in results["metadatas"] if meta and "source_file" in meta)
    
    all_metrics = []
    
    # 2. Loop through EACH file and extract its specific metrics
    for filename in unique_files:
        query_embed = client.models.embed_content(
            model='gemini-embedding-001',
            contents="financial summary total revenue net income EPS forward guidance"
        ).embeddings[0].values
        
        # TARGETED RAG: Only search chunks belonging to this specific file
        file_results = collection.query(
            query_embeddings=[query_embed],
            n_results=5,
            where={"$and": [{"user_id": user_id}, {"source_file": filename}]}
        )
        
        if file_results['documents'] and file_results['documents'][0]:
            context = "\n\n".join(file_results['documents'][0])
            
            json_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinancialMetrics,
                temperature=0.0
            )
            
            # Tell the AI exactly which file it is looking at so it fills out the schema correctly
            prompt = f"Extract the core financial metrics for the company in this context. The source_file is '{filename}':\n\n{context}"
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=json_config
            )
            
            import json
            try:
                # Add the extracted JSON object to our master list
                all_metrics.append(json.loads(response.text))
            except Exception:
                pass
                
    return all_metrics

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
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(code_execution=types.ToolCodeExecution()) # <-- The Math Agent
        ],
        thinking_config=types.ThinkingConfig(
            include_thoughts=True # <-- The Reasoning Agent
        ),
        temperature=0.0 # Keeps the model highly factual
    )
    
    # 5. Generate Answer
    final_answer = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=final_prompt,
        config=agent_config
    )
    
    # 6. Parse the Agent's multi-part response
    answer_text = ""
    thoughts_text = ""
    code_data = []
    
    try:
        for part in final_answer.candidates[0].content.parts:
            # Extract Reasoning (Thoughts)
            if getattr(part, 'thought', False) and part.text:
                thoughts_text += part.text + "\n\n"
            # Extract the actual text answer
            elif part.text:
                answer_text += part.text
                
            # Extract Python Code written by the AI
            if getattr(part, 'executable_code', None):
                code_data.append({
                    "type": "code",
                    "code": part.executable_code.code
                })
            # Extract the Terminal Output of that code
            if getattr(part, 'code_execution_result', None):
                code_data.append({
                    "type": "result",
                    "output": part.code_execution_result.output
                })
    except Exception:
        answer_text = final_answer.text # Safety fallback
        
    # 7. Extract Web Search Metadata
    search_queries = []
    try:
        queries = final_answer.candidates[0].grounding_metadata.web_search_queries
        if queries:
            search_queries = list(queries)
    except Exception:
        pass

    # 8. Save EVERYTHING to TinyDB
    final_clean_answer = answer_text.strip() or final_answer.text
    chat_collection.insert({
        "user_id": user_id, 
        "role": "assistant", 
        "content": final_clean_answer,
        "thoughts": thoughts_text.strip(),
        "code_execution": code_data,
        "search_queries": search_queries
    })
        
    return {
        "answer": final_clean_answer,
        "search_queries": search_queries,
        "thoughts": thoughts_text.strip(),
        "code_execution": code_data
    }
    


# ---------------------------------------------------------
# ENDPOINT 4: Get Chat History
# ---------------------------------------------------------
@app.get("/api/history/{user_id}")
def get_chat_history(user_id: str):
    MessageQuery = Query()
    user_docs = chat_collection.search(MessageQuery.user_id == user_id)
    return [{
        "role": doc["role"], 
        "content": doc["content"],
        "thoughts": doc.get("thoughts", ""),
        "code_execution": doc.get("code_execution", []),
        "search_queries": doc.get("search_queries", [])
    } for doc in user_docs]