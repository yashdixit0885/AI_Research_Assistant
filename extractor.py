import streamlit as st
import PyPDF2
from google import genai
import os
from dotenv import load_dotenv
import chromadb

# Unlock the vault
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=my_key)

# ---------------------------------------------------------
# NEW: Initialize ChromaDB
# This creates a folder named "pdf_db" in your project directory
chroma_client = chromadb.PersistentClient(path="./pdf_db")
# ---------------------------------------------------------

st.title("📈 AI Financial Researcher")
st.write("Upload a financial PDF (like an Earnings Report).")

# 1. Build the PDF Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # 2. Read the PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() + "\n\n"
        
    # 3. Robust Overlapping Chunking
    chunk_size = 1000
    overlap = 200
    text_chunks = []
    
    for i in range(0, len(raw_text), chunk_size - overlap):
        chunk = raw_text[i : i + chunk_size].strip()
        if len(chunk) > 100:
            text_chunks.append(chunk)

    st.success(f"Successfully split the PDF into {len(text_chunks)} overlapping chunks!")
    
    # 4. Create the Vector Database Index
    if st.button("Build AI Index"):
        with st.spinner("Generating embeddings and saving to ChromaDB..."):
            
            # Reset the collection so we don't get duplicates if you click twice
            try:
                chroma_client.delete_collection("earnings_reports")
            except Exception as e:
                pass # It's fine if the collection doesn't exist yet
                
            collection = chroma_client.create_collection(name="earnings_reports")
            
            embeddings_list = []
            ids_list = []
            
            # Embed each chunk
            for i, chunk in enumerate(text_chunks):
                response = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=chunk
                )
                embeddings_list.append(response.embeddings[0].values)
                # ChromaDB requires a unique ID for every single chunk
                ids_list.append(f"chunk_{i}")
            
            # Add everything to the permanent database in one go
            collection.add(
                documents=text_chunks,
                embeddings=embeddings_list,
                ids=ids_list
            )
            
            st.success("Index built and permanently saved to your local database! 🗄️")

st.divider()
st.subheader("🔍 Ask the Document")
user_question = st.text_input("What would you like to know about this financial report?")

if st.button("Search") and user_question:
    # Safely connect to the collection
    collection = chroma_client.get_or_create_collection(name="earnings_reports")
    
    if collection.count() > 0:
        with st.spinner("Searching database and analyzing..."):
            
            # 1. Embed the user's question
            question_embedding = client.models.embed_content(
                model='gemini-embedding-001',
                contents=user_question
            ).embeddings[0].values

            # 2. Query ChromaDB directly (No more manual math loops!)
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3 # Get the top 3 matches
            )
            
            # ChromaDB returns a dictionary; extract the document text
            top_chunks = results['documents'][0]
            combined_context = "\n\n---\n\n".join(top_chunks)
            
            # 3. The Professional Analyst Prompt
            final_prompt = f"""
            You are a Senior Financial Research Analyst. Your goal is to provide accurate, 
            evidence-based answers derived ONLY from the provided context.

            ### INSTRUCTIONS:
            1. START with a "Bottom Line" section: A 1-2 sentence direct answer to the question.
            2. FOLLOW with a "Supporting Details" section: Use bullet points to extract specific metrics, dates, or commentary from the context.
            3. CONCLUDE with "Source Reference": Quote the specific sentence or phrase from the context that contains the primary answer.
            4. If the answer is not in the context, state: "Information not available in the provided document segments."

            ### CONTEXT:
            {combined_context}

            ### USER QUESTION:
            {user_question}
            """
            
            # 4. Generate and display the answer
            final_answer = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=final_prompt
            )
            st.markdown(final_answer.text)
            
            # Verify the DB retrieval worked
            with st.expander("See the exact source text retrieved from the DB"):
                st.write(combined_context)
    else:
        st.warning("Please upload a PDF and build the index first.")