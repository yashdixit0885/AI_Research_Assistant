import streamlit as st
import requests
import pandas as pd
import io

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
# PDF UPLOAD & VAULT MANAGEMENT
# ---------------------------------------------------------
st.sidebar.subheader("📂 Document Management")

# 1. Fetch active documents from the backend API
try:
    docs_response = requests.get(f"{API_BASE_URL}/documents/{current_user}")
    active_docs = docs_response.json().get("documents", []) if docs_response.status_code == 200 else []
except requests.exceptions.ConnectionError:
    active_docs = []
    st.sidebar.error("🚨 API disconnected. Is FastAPI running?")

# 2. Display the vault contents and delete controls
if active_docs:
    st.sidebar.write("**Active Vault Documents:**")
    for doc in active_docs:
        col1, col2 = st.sidebar.columns([4, 1]) # 4 parts text, 1 part button
        with col1:
            st.write(f"📄 {doc}")
        with col2:
            # Individual delete button
            if st.button("❌", key=f"del_{doc}", help=f"Remove {doc}"):
                requests.delete(f"{API_BASE_URL}/documents/{current_user}/{doc}")
                st.rerun() # Instantly refresh the UI
    
    st.sidebar.write("---")
    # Master clear button
    if st.sidebar.button("🗑️ Clear Entire Vault", type="primary", use_container_width=True):
        requests.delete(f"{API_BASE_URL}/documents/{current_user}")
        st.rerun()
else:
    st.sidebar.info("⚠️ Your vault is empty. Please upload a PDF.")

# 3. The Uploader
uploaded_file = st.sidebar.file_uploader("Upload a new PDF to your vault", type="pdf", key=f"pdf_uploader_{current_user}")

if uploaded_file is not None:
    if st.sidebar.button("➕ Add to Vault", use_container_width=True):
        with st.spinner("Sending document to backend API..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_BASE_URL}/upload/{current_user}", files=files)
            
            if response.status_code == 200:
                st.sidebar.success(f"✅ {response.json()['message']}")
                st.rerun() # Instantly refresh the UI to show the new file
            else:
                st.sidebar.error("Failed to process document.")

st.sidebar.divider()

# ---------------------------------------------------------
# EXCEL METRICS EXTRACTION
# ---------------------------------------------------------
st.sidebar.subheader("📊 Financial Reports")

# 1. The Generation Button
if st.sidebar.button("⚙️ Generate Summary Excel", use_container_width=True):
    with st.sidebar.status("Analyzing documents and extracting metrics..."):
        # HTTP GET: Request the structured JSON from our API
        response = requests.get(f"{API_BASE_URL}/extract/{current_user}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert the JSON dictionary into a Pandas DataFrame (a table)
            df = pd.DataFrame(data)
            
            # Create an in-memory Excel file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Summary Metrics')
            
            # Save the raw Excel bytes to Streamlit's session state
            st.session_state[f"excel_file_{current_user}"] = buffer.getvalue()
            st.sidebar.success("✅ Excel file generated!")
        else:
            st.sidebar.error("Failed to extract metrics.")

# 2. The Download Button (Only shows up AFTER generation is complete)
if f"excel_file_{current_user}" in st.session_state:
    st.sidebar.download_button(
        label="📥 Download Excel File",
        data=st.session_state[f"excel_file_{current_user}"],
        file_name=f"Financial_Summary_{current_user}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------
st.divider()
st.subheader(f"💬 Chatting as {current_user}")

# 1. Render previous messages AND their expanders
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("thoughts"):
            with st.expander("🧠 See Agent Reasoning"):
                st.write(message["thoughts"])
                
        if message.get("code_execution"):
            with st.expander("💻 See Code Execution"):
                for step in message["code_execution"]:
                    if step["type"] == "code":
                        st.markdown(f"**Generated Code:**\n```python\n{step['code']}\n```")
                    elif step["type"] == "result":
                        st.markdown(f"**Execution Output:**\n```text\n{step['output']}\n```")
                        
        if message.get("search_queries"):
            with st.expander("🌐 See Web Search Queries"):
                for q in message["search_queries"]:
                    st.write(f"🔍 {q}")

# 2. Handle New User Questions
if user_question := st.chat_input("Ask a question about the report or live market..."):
    with st.chat_message("user"):
        st.markdown(user_question)
        
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("API is analyzing PDF, writing code, and browsing the web..."):
            
            payload = {"question": user_question}
            response = requests.post(f"{API_BASE_URL}/chat/{current_user}", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                thoughts = data.get("thoughts", "")
                code_execution = data.get("code_execution", [])
                search_queries = data.get("search_queries", [])
                
                # Display the primary answer
                st.markdown(answer)
                
                # Render the collapsible transparency blocks
                if thoughts:
                    with st.expander("🧠 See Agent Reasoning"):
                        st.write(thoughts)
                if code_execution:
                    with st.expander("💻 See Code Execution"):
                        for step in code_execution:
                            if step["type"] == "code":
                                st.markdown(f"**Generated Code:**\n```python\n{step['code']}\n```")
                            elif step["type"] == "result":
                                st.markdown(f"**Execution Output:**\n```text\n{step['output']}\n```")
                if search_queries:
                    with st.expander("🌐 See Web Search Queries"):
                        for q in search_queries:
                            st.write(f"🔍 {q}")

                # Save EVERYTHING to Streamlit's temporary memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "thoughts": thoughts,
                    "code_execution": code_execution,
                    "search_queries": search_queries
                })
            else:
                st.error("The API encountered an error processing your request.")