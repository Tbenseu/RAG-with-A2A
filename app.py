import streamlit as st
import os
import json
import hashlib
import grpc
import asyncio
from datetime import datetime
from qa_system_with_a2a_demo.tools.document_tool import DocumentTool  # Your existing tool
from qa_system_with_a2a_demo.agents.orchestrator.main import OrchestratorAgent
# orchestrator_agent import OrchestratorAgent  # Your orchestrator
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2_grpc
from qa_system_with_a2a_demo.agents.rag_agent import rag_agent_pb2
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2
from qa_system_with_a2a_demo.agents.web_search_agent import web_search_agent_pb2_grpc
from qa_system_with_a2a_demo.agents.orchestrator import orchestrator_agent_pb2
from qa_system_with_a2a_demo.agents.orchestrator import orchestrator_agent_pb2_grpc
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict
import json
import logging
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def create_orchestrator_stub():
#     channel = grpc.insecure_channel('localhost:8000')
#     return orchestrator_agent_pb2_grpc.OrchestratorServiceStub(channel)

async def get_orchestrator_response(prompt: str) -> dict:
    async with grpc.aio.insecure_channel('localhost:8000') as channel:
        stub = orchestrator_agent_pb2_grpc.OrchestratorServiceStub(channel)
        request = orchestrator_agent_pb2.OrchestrationRequest(
            query=prompt,
            context={"chat_id": st.session_state.current_chat}
        )
        response = await stub.ProcessQuery(request)
        # Convert the Struct to a Python dict
        response_dict = MessageToDict(response.response)
        
        # # Extract the text (assuming it's stored under 'text' key in the Struct)
        text = response_dict.get('response', '').get('text', '')

        print(f"Response >> {response_dict}")
        # print(json.loads(response_dict))
        # return json.loads(response_dict.response)
        return response_dict

# Initialize components
document_tool = DocumentTool()
orchestrator = OrchestratorAgent()

# User management functions
def init_user_db():
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)

def create_user(username, password):
    with open('users.json', 'r') as f:
        users = json.load(f)
    
    if username in users:
        return False
    
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    users[username] = {
        'salt': salt.hex(),
        'key': key.hex(),
        'chats': {}
    }
    
    with open('users.json', 'w') as f:
        json.dump(users, f)
    return True

def verify_user(username, password):
    with open('users.json', 'r') as f:
        users = json.load(f)
    
    if username not in users:
        return False
    
    user_data = users[username]
    salt = bytes.fromhex(user_data['salt'])
    key = bytes.fromhex(user_data['key'])
    
    new_key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return new_key == key

# Session state management
def init_session_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# Authentication UI
def show_auth():
    st.title("Document Chat System")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if verify_user(username, password):
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                if new_pass != confirm_pass:
                    st.error("Passwords don't match")
                elif create_user(new_user, new_pass):
                    st.success("Account created! Please login")
                else:
                    st.error("Username already exists")

# Chat management UI
def show_chat_selector():
    with open('users.json', 'r') as f:
        user_data = json.load(f).get(st.session_state.user, {})
    
    st.sidebar.title(f"Welcome, {st.session_state.user}")
    
    # Create new chat
    new_chat_name = st.sidebar.text_input("New chat name")
    if st.sidebar.button("Create Chat"):
        if new_chat_name:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            chat_id = f"{timestamp}_{new_chat_name}"
            user_data['chats'][chat_id] = {"name": new_chat_name, "messages": []}
            
            with open('users.json', 'r+') as f:
                users = json.load(f)
                users[st.session_state.user] = user_data
                f.seek(0)
                json.dump(users, f)
            
            st.session_state.current_chat = chat_id
            st.session_state.messages = []
            st.rerun()
    
    # List existing chats
    st.sidebar.subheader("Your Chats")
    for chat_id, chat_info in user_data.get('chats', {}).items():
        if st.sidebar.button(chat_info['name'], key=chat_id):
            st.session_state.current_chat = chat_id
            st.session_state.messages = chat_info.get('messages', [])
            st.rerun()

# Document processing
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            # Create RAG client
            channel = grpc.insecure_channel('localhost:8001')
            rag_stub = rag_agent_pb2_grpc.RagAgentServiceStub(channel)
            # Send file to RAG agent
            response = rag_stub.ProcessDocument(
                rag_agent_pb2.FileUpload(
                    content=uploaded_file.read(),
                    filename=uploaded_file.name
                )
            )
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

            # with st.spinner("Processing document..."):
            #     document_tool.handle_uploaded_file(uploaded_file)
            #     st.success("Document processed and ready for queries!")

# Chat UI
def show_chat():
    st.title(st.session_state.current_chat.split('_', 1)[1])
    
    # Document uploader
    uploaded_file = st.file_uploader(
        "Upload a document for this chat",
        type=['pdf'],
        key=f"uploader_{st.session_state.current_chat}"
    )
    process_uploaded_file(uploaded_file)
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run async code in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response_dict = loop.run_until_complete(
                        get_orchestrator_response(prompt)
                    )
                    # print(response_dict)

                
                    
                    if "response" in response_dict:
                        answer = response_dict["response"].get("text", "")
                        st.markdown(answer)
                        
                        # if "sources" in response_dict:
                        #     st.markdown("**Sources:**")
                        #     for src in response_dict["sources"]:
                        #         st.markdown(f"- {src}")
                                
                        # if "comparison" in response_dict:
                        #     with st.expander("Response Analysis"):
                        #         st.json(response_dict["comparison"])
                    else:
                        st.error("Unexpected response format from orchestrator")
                    
                    # Store the successful response
                    answer = answer if 'answer' in locals() else str(response_dict)
                
                except Exception as e:
                    st.error(f"Error communicating with orchestrator: {str(e)}")
                    answer = f"Error: {str(e)}"

        
        # Add AI response to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer #if 'answer' in locals() else str(response)
        })
        
        # Save chat history
        with open('users.json', 'r+') as f:
            users = json.load(f)
            users[st.session_state.user]['chats'][st.session_state.current_chat]['messages'] = st.session_state.messages
            f.seek(0)
            json.dump(users, f, indent=2)

# Main app flow
def main():
    init_user_db()
    init_session_state()
    
    if st.session_state.user is None:
        show_auth()
    else:
        if st.session_state.current_chat is None:
            show_chat_selector()
        else:
            show_chat()

if __name__ == "__main__":
    main()