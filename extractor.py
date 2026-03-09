import streamlit as st
import requests

# Point to your local FastAPI server
API_BASE_URL = "http://127.0.0.1:8000/api"

st.title("📈 AI Financial Researcher")
st.write("Ask questions combining your PDF report with live market data.")

# ---------------------------------------------------------
# MULTI-TENANT AUTHENTICATION (MOCK)
# ---------------------------------------------------------
st.sidebar.title("🔐 Authentication")
current_user = st.sidebar.selectbox("Logged in as:", ["User_Alpha", "User_Beta", "User_Gamma"])

# State Management: Fetch history from API when switching users
if "active_user" not in st.session_state or st.session_state.active_user != current_user:
    st.session_state.active_user = current_user
    st.session_state.messages = [] 
    
    # HTTP GET: Fetch chat history from the backend
    try:
        response = requests.get(f"{API_BASE_URL}/history/{current_user}")
        if response.status_code == 200:
            st.session_state.messages = response.json()
    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to the API backend. Is FastAPI running?")

# ---------------------------------------------------------
# PDF UPLOAD & JSON EXTRACTION
# ---------------------------------------------------------
st.sidebar.subheader("📂 Document Management")

uploaded_file = st.sidebar.file_uploader("Upload a new PDF to your vault", type="pdf", key=f"pdf_uploader_{current_user}")

if uploaded_file is not None:
    if st.sidebar.button("Build AI Index"):
        with st.spinner("Sending document to backend API..."):
            # HTTP POST: Send the file to the backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/upload/{current_user}", files=files)
            
            if response.status_code == 200:
                st.sidebar.success(f"✅ {response.json()['message']}")
            else:
                st.sidebar.error("Failed to process document.")

st.sidebar.divider()

if st.sidebar.button("📊 Extract JSON Metrics"):
    with st.sidebar.status("Fetching structured data from API..."):
        # HTTP GET: Request the structured JSON extraction
        response = requests.get(f"{API_BASE_URL}/extract/{current_user}")
        
        if response.status_code == 200:
            st.sidebar.json(response.json())
        else:
            st.sidebar.error(response.json().get("detail", "Failed to extract metrics."))

# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------
st.divider()
st.subheader(f"💬 Chatting as {current_user}")

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_question := st.chat_input("How can I help you today?"):
    
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_question)
        
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("API is analyzing PDF and browsing the live web..."):
            
            # HTTP POST: Send the question to the backend chat agent
            payload = {"question": user_question}
            response = requests.post(f"{API_BASE_URL}/chat/{current_user}", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                search_queries = data.get("search_queries", [])
                
                # Display the AI's response
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Display search metadata if the backend used the web tool
                if search_queries:
                    with st.expander("🌐 See Web Search Queries"):
                        for q in search_queries:
                            st.write(f"🔍 {q}")
            else:
                st.error("The API encountered an error processing your request.")