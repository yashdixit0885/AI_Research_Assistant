import streamlit as st
import PyPDF2
from google import genai
import os
from dotenv import load_dotenv

def calculate_dot_product(vec1, vec2):
    # This multiplies the coordinates and adds them up to find the closest match
    return sum(a * b for a, b in zip(vec1, vec2))
# Unlock the vault
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=my_key)

st.title("📈 AI Financial Researcher")
st.write("Upload a financial PDF (like an Earnings Report).")

# 1. Build the PDF Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # 2. Read the PDF using PyPDF2
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        # Extract text from each page and add a double newline at the end
        raw_text += page.extract_text() + "\n\n"
        
    # 3. Chop the text into chunks
    # We split by the double newline as we discussed
    # 3. Robust Overlapping Chunking
    chunk_size = 1000
    overlap = 200
    text_chunks = []
    
    # We loop through the text, jumping forward by (chunk_size - overlap) each time
    for i in range(0, len(raw_text), chunk_size - overlap):
        chunk = raw_text[i : i + chunk_size]
        text_chunks.append(chunk.strip())
    
    # Clean up: remove any tiny fragments at the very end
    text_chunks = [c for c in text_chunks if len(c) > 100]

    st.success(f"Successfully split the PDF into {len(text_chunks)} overlapping chunks!")
    
    # 4. Create the in-memory index
    # We use st.session_state so the index isn't deleted when the page refreshes
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = []

    if st.button("Build AI Index"):
        with st.spinner("Generating embeddings..."):
            # Clear the index in case the user clicks the button twice
            st.session_state.vector_index = [] 
            
            for chunk in text_chunks:
                # Ask Gemini to create the mathematical embedding
                response = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=chunk
                )
                
                # Store the text and its mathematical coordinates together
                st.session_state.vector_index.append({
                    "text": chunk,
                    "embedding": response.embeddings[0].values
                })
            
            st.success("Index built! The document is ready to be searched.")# We will add the embedding step here next!
st.divider()
st.subheader("🔍 Ask the Document")
user_question = st.text_input("What would you like to know about this financial report?")

if st.button("Search") and user_question:
    if "vector_index" in st.session_state and len(st.session_state.vector_index) > 0:
        with st.spinner("Searching and analyzing..."):
            # 1. Embed the user's question
            question_embedding = client.models.embed_content(
                model='gemini-embedding-001',
                contents=user_question
            ).embeddings[0].values

            # 2. Score all chunks
            scored_chunks = []
            for item in st.session_state.vector_index:
                score = calculate_dot_product(question_embedding, item["embedding"])
                scored_chunks.append({"text": item["text"], "score": score})
            
            # 3. Sort by score (highest first) and take the Top 3
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            top_chunks = scored_chunks[:3]
            
            # Combine the text from the top 3 chunks into one big context string
            combined_context = "\n\n---\n\n".join([c["text"] for c in top_chunks])
            
            # 4. Context Injection with multiple chunks
            final_prompt = f"""
                You are a Senior Financial Research Analyst. Your goal is to provide accurate, 
                evidence-based answers derived ONLY from the provided context.

                ### INSTRUCTIONS:
                1. START with a "Bottom Line" section: A 1-2 sentence direct answer to the question.
                2. FOLLOW with a "Supporting Details" section: Use bullet points to extract specific 
                metrics, dates, or commentary from the context.
                3. CONCLUDE with "Source Reference": Quote the specific sentence or phrase from the 
                context that contains the primary answer.
                4. If the answer is not in the context, state: "Information not available in the provided document segments."

                ### CONTEXT:
                {combined_context}

                ### USER QUESTION:
                {user_question}
                """
            
            # 5. Generate and display the answer
            final_answer = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=final_prompt
            )
            
            
            # Display the answer
            st.markdown(final_answer.text)
            
            # Bonus: Let the user peek at the raw chunk we found to verify the math worked
            with st.expander("See the exact source text retrieved from the PDF"):
                st.write(top_chunks)
    else:
        st.warning("Please upload a PDF and build the index first.")