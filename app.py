"""
Streamlit Chat Interface for University RAG Assistant
Beautiful web UI for the chatbot
"""

import streamlit as st
from src.rag_pipeline import UniversityRAG
import time
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Conestoga College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG system (cached for performance)
@st.cache_resource
def initialize_rag():
    """Initialize and cache the RAG system"""
    with st.spinner("Initializing AI Assistant..."):
        rag = UniversityRAG(
            vectorstore_path="data/vectorstore",
            collection_name="university_knowledge",
            llm_model="llama3.2:3b",
            top_k=5
        )
    return rag


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = initialize_rag()


# Sidebar
with st.sidebar:
    st.markdown("###Conestoga Assistant")
    st.markdown("---")
    
    # Information
    st.markdown("#### About")
    st.info(
        "I'm your virtual assistant for Conestoga College. "
        "Ask me anything about programs, admissions, student services, and more!"
    )
    
    # Statistics
    st.markdown("#### System Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Knowledge Base", f"{st.session_state.rag_system.collection.count()} chunks")
    with col2:
        st.metric("Conversations", len(st.session_state.messages) // 2)
    
    # Settings
    st.markdown("#### Settings")
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="More sources = better coverage but slower responses"
    )
    st.session_state.rag_system.top_k = top_k
    
    show_sources = st.checkbox("Show sources", value=True)
    show_metrics = st.checkbox("Show performance metrics", value=False)
    
    # Clear conversation
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Sample questions
    st.markdown("#### Try asking:")
    sample_questions = [
        "What programs are offered?",
        "What are admission requirements?",
        "Tell me about student services",
        "How do I apply?",
        "What is the tuition cost?"
    ]
    
    for question in sample_questions:
        if st.button(f"'{question}'", key=question, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()


# Main content
st.markdown('<div class="main-header">🎓 Conestoga College Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about the college!</div>', unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and show_sources:
                if message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{i}. {source['title']}</strong><br>
                                <a href="{source['url']}" target="_blank">{source['url']}</a><br>
                                <small>Relevance: {source['relevance']:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Display metrics if available and enabled
            if message["role"] == "assistant" and "metrics" in message and show_metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval", f"{message['metrics']['retrieval_time']:.2f}s")
                with col2:
                    st.metric("Generation", f"{message['metrics']['generation_time']:.2f}s")
                with col3:
                    st.metric("Total", f"{message['metrics']['total_time']:.2f}s")

# Chat input
if prompt := st.chat_input("Ask me anything about Conestoga College..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response from RAG system
            result = st.session_state.rag_system.query(prompt, verbose=False)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Display sources
            if show_sources and result['sources']:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{i}. {source['title']}</strong><br>
                            <a href="{source['url']}" target="_blank">{source['url']}</a><br>
                            <small>Relevance: {source['relevance']:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display metrics
            if show_metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval", f"{result['retrieval_time']:.2f}s")
                with col2:
                    st.metric("Generation", f"{result['generation_time']:.2f}s")
                with col3:
                    st.metric("Total", f"{result['total_time']:.2f}s")
    
    # Save assistant message to history
    assistant_message = {
        "role": "assistant",
        "content": result['answer'],
        "sources": result['sources'],
        "metrics": {
            "retrieval_time": result['retrieval_time'],
            "generation_time": result['generation_time'],
            "total_time": result['total_time']
        }
    }
    st.session_state.messages.append(assistant_message)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "💡 Powered by RAG (Retrieval-Augmented Generation) | "
    "Built with Streamlit, ChromaDB & Ollama | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
    "</div>",
    unsafe_allow_html=True
)