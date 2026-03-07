import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Unlock the vault
load_dotenv()
my_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=my_key)

st.title("🌐 Live Market Web Agent")
st.write("Ask a question about real-time market data or breaking news.")

user_question = st.text_input("What do you want to know?")

if st.button("Search Web") and user_question:
    with st.spinner("Browsing the live web..."):
        
        # 1. Define the Google Search Tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # 2. Add the tool to the generation configuration
        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )

        # 3. Ask the model, passing the config toolbox
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_question,
            config=config
        )
        
        # 4. Display the answer
        st.markdown(response.text)
        
        # Bonus: Let's try to peek at the exact Google search queries the AI decided to run!
        with st.expander("See the background search queries"):
            try:
                queries = response.candidates[0].grounding_metadata.web_search_queries
                for q in queries:
                    st.write(f"🔍 {q}")
            except Exception:
                st.write("No external queries were generated.")