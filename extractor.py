import streamlit as st
import PyPDF2
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import chromadb
from tinydb import TinyDB, Query

# Unlock the vault
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=my_key)

# ---------------------------------------------------------
# DATABASES INITIALIZATION
# ---------------------------------------------------------
# 1. Vector Database (ChromaDB) for PDFs
chroma_client = chromadb.PersistentClient(path="./pdf_db")

# 2. NoSQL Database (TinyDB) for Chat History
db = TinyDB('nosql_chat_db.json')
chat_collection = db.table('messages')

st.title("📈 AI Financial Researcher")
st.write("Ask questions combining your PDF report with live market data.")

# ---------------------------------------------------------
# MULTI-TENANT AUTHENTICATION (MOCK)
# ---------------------------------------------------------
st.sidebar.title("🔐 Authentication")
st.sidebar.write("Simulate multi-tenant login:")
current_user = st.sidebar.selectbox("Logged in as:", ["User_Alpha", "User_Beta", "User_Gamma"])

# State Management: Detect if the user switched accounts
if "active_user" not in st.session_state or st.session_state.active_user != current_user:
    st.session_state.active_user = current_user
    st.session_state.messages = [] # Clear UI memory
    
    # Query TinyDB: Retrieve ONLY the current user's documents
    MessageQuery = Query()
    user_docs = chat_collection.search(MessageQuery.user_id == current_user)
    
    # Hydrate the UI with the database results
    for doc in user_docs:
        st.session_state.messages.append({"role": doc["role"], "content": doc["content"]})

# ---------------------------------------------------------
# PDF UPLOAD & INDEXING
# ---------------------------------------------------------
st.sidebar.subheader("📂 Document Management")

# 1. Check if the user already has data in the database
collection = chroma_client.get_or_create_collection(name="earnings_reports")
user_existing_data = collection.get(where={"user_id": current_user})

# 2. Show a status indicator so the user knows they don't NEED to upload again
if user_existing_data and len(user_existing_data['ids']) > 0:
    # Grab the filename from the metadata of the very first chunk
    # We use .get() as a safety net in case older chunks don't have this key
    first_chunk_metadata = user_existing_data['metadatas'][0]
    saved_filename = first_chunk_metadata.get("source_file", "an existing document")
    
    st.sidebar.success(f"✅ Active document in vault: **{saved_filename}**")
    st.sidebar.write("*(Upload a new PDF below to replace it)*")
else:
    st.sidebar.info("⚠️ Your vault is empty. Please upload a PDF.")

# 3. The Uploader
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", key=f"pdf_uploader_{current_user}")

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() + "\n\n"
        
    chunk_size = 1000
    overlap = 200
    text_chunks = []
    
    for i in range(0, len(raw_text), chunk_size - overlap):
        chunk = raw_text[i : i + chunk_size].strip()
        if len(chunk) > 100:
            text_chunks.append(chunk)

    st.sidebar.success(f"Successfully split PDF into {len(text_chunks)} chunks!")
    
    if st.sidebar.button("Build AI Index"):
        with st.spinner("Generating embeddings and saving to ChromaDB..."):
            
            # 1. Connect to the collection (DO NOT delete the whole thing anymore)
            collection = chroma_client.get_or_create_collection(name="earnings_reports")
            
            # 2. Delete ONLY the old chunks belonging to the current user
            try:
                collection.delete(where={"user_id": current_user})
            except Exception:
                pass # It's fine if they don't have any old chunks yet
            
            embeddings_list = []
            ids_list = []
            metadatas_list = [] # NEW: We need a list for our nametags
            
            for i, chunk in enumerate(text_chunks):
                response = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=chunk
                )
                embeddings_list.append(response.embeddings[0].values)
                
                # NEW: Make the ID completely unique to the user to prevent overwriting
                ids_list.append(f"{current_user}_chunk_{i}")
                
                # NEW: Create the nametag
                metadatas_list.append({"user_id": current_user,"source_file": uploaded_file.name})
            
            # 3. Add to database with the metadatas included
            collection.add(
                documents=text_chunks,
                embeddings=embeddings_list,
                metadatas=metadatas_list, # <--- The magic nametags!
                ids=ids_list
            )
            
            st.sidebar.success("Index saved to your securely isolated vault! 🗄️")

# ---------------------------------------------------------
# CHAT INTERFACE & AGENT LOGIC
# ---------------------------------------------------------
st.divider()
st.subheader(f"💬 Chatting as {current_user}")

# Render all previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if user_question := st.chat_input("How can I help you today?"):
    
    # 1. Display & Save User Message
    with st.chat_message("user"):
        st.markdown(user_question)
        
    st.session_state.messages.append({"role": "user", "content": user_question})
    chat_collection.insert({"user_id": current_user, "role": "user", "content": user_question}) # Save to TinyDB

    # Connect to ChromaDB
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    if collection.count() > 0:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing PDF and browsing the live web..."):
                
                # 2. Retrieve relevant PDF chunks
                question_embedding = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=user_question
                ).embeddings[0].values

                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=3,
                    where={"user_id": current_user} 
                )
                
                if len(results['documents'][0]) > 0:
                    top_chunks = results['documents'][0]
                    combined_context = "\n\n---\n\n".join(top_chunks)
                else:
                    combined_context = "No PDF documents uploaded by this user yet."
                
                # 3. Build Conversation Memory String
                conversation_string = ""
                for msg in st.session_state.messages[:-1]: 
                    speaker = "User" if msg["role"] == "user" else "Assistant"
                    conversation_string += f"{speaker}: {msg['content']}\n\n"

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
                {user_question}
                """
                
                agent_config = types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
                
                # 5. Generate and Display Answer
                final_answer = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=final_prompt,
                    config=agent_config
                )
                
                st.markdown(final_answer.text)
                
                # 6. Save Assistant Message
                st.session_state.messages.append({"role": "assistant", "content": final_answer.text})
                chat_collection.insert({"user_id": current_user, "role": "assistant", "content": final_answer.text}) # Save to TinyDB
                
                # Optional: Show search metadata expander
                with st.expander("🌐 See Web Search Queries"):
                    try:
                        queries = final_answer.candidates[0].grounding_metadata.web_search_queries
                        for q in queries:
                            st.write(f"🔍 {q}")
                    except Exception:
                        st.write("No external queries were generated for this prompt.")
    else:
        st.error("⚠️ Please upload a PDF and click 'Build AI Index' first.")