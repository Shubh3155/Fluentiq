import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()  # Loads .env file

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_VECTORSTORE_NAME = "faiss_vectorstore"  # Change this to your vectorstore name

# Streamlit Config
st.set_page_config(page_title="Fluentiq RAG", page_icon="🗣️", layout="wide")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
if "vectorstore_name" not in st.session_state:
    st.session_state.vectorstore_name = DEFAULT_VECTORSTORE_NAME

class RAGChatbot:
    def __init__(self, google_api_key):
        self.api_key = google_api_key
        genai.configure(api_key=self.api_key)
        self.embeddings = None
        self.chat_model = None
        self._initialize_embeddings()
        self._initialize_chat_model()
    
    def _initialize_embeddings(self):
        """Initialize Google embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        except Exception as e:
            st.error(f"Error initializing embeddings: {e}")
    
    def _initialize_chat_model(self):
        """Initialize Gemini chat model"""
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "You are Fluentiq, a friendly and knowledgeable RAG-powered AI assistant. "
                    "When provided with document context, use it as your primary source of information. "
                    "Always indicate when you're using information from uploaded documents vs. general knowledge. "
                    "If the document context doesn't contain relevant information, say so and provide general knowledge. "
                    "Be conversational, helpful, and accurate. "
                    "Prioritize document-based information over general knowledge when available."
                )
            )
            self.chat_model = model.start_chat()
        except Exception as e:
            st.error(f"Error initializing chat model: {e}")
    
    def load_vectorstore(self, vectorstore_name):
        """Load existing FAISS vectorstore"""
        try:
            if not os.path.exists(vectorstore_name):
                return None, f"Vectorstore directory '{vectorstore_name}' not found"
            
            vectorstore = FAISS.load_local(
                vectorstore_name, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return vectorstore, "Successfully loaded vectorstore"
        except Exception as e:
            return None, f"Error loading vectorstore: {e}"
    
    def retrieve_relevant_chunks(self, query, vectorstore, k=3):
        """Retrieve relevant document chunks using RAG"""
        try:
            if vectorstore is None:
                return []
            
            # Perform similarity search
            retrieved_docs = vectorstore.similarity_search(query, k=k)
            return retrieved_docs
            
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            return []
    
    def search_web(self, query):
        """Simple web search function as fallback"""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search snippets
            snippets = []
            for result in soup.find_all('div', class_='BNeawe')[:3]:
                if result.text:
                    snippets.append(result.text)
            
            return " ".join(snippets) if snippets else "No relevant information found on the web."
            
        except Exception as e:
            return f"Web search temporarily unavailable: {str(e)[:100]}"
    
    def generate_response(self, user_input, vectorstore):
        """Generate response using RAG"""
        try:
            context_info = ""
            
            # RAG: Retrieve relevant chunks from vectorstore
            if vectorstore:
                retrieved_docs = self.retrieve_relevant_chunks(user_input, vectorstore, k=3)
                
                if retrieved_docs:
                    doc_contexts = []
                    for i, doc in enumerate(retrieved_docs, 1):
                        doc_contexts.append(f"Document Chunk {i}:\n{doc.page_content}")
                    
                    context_info = "Retrieved Context from Your Document:\n" + "\n\n".join(doc_contexts) + "\n\n"
            
            # Fallback to web search if no document context available
            if not context_info:
                web_results = self.search_web(user_input)
                if web_results and "No relevant information found" not in web_results:
                    context_info = f"Web Search Results:\n{web_results}\n\n"
            
            # Create the RAG prompt
            if context_info.strip():
                prompt = f"""Context Information:
{context_info}

User Question: {user_input}

Instructions: 
- Use the provided context as your primary source of information
- If using document context, mention that you're referencing the uploaded document
- If using web search results, mention that you're using current web information
- If the context doesn't fully answer the question, supplement with your general knowledge but clearly distinguish between sources
- Provide a comprehensive, natural response

Response:"""
            else:
                prompt = f"""User Question: {user_input}

Please provide a helpful response based on your general knowledge. Be conversational and informative."""
            
            # Generate response
            response = self.chat_model.send_message(prompt)
            return response.text
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)[:100]}... Please try again!"

# Initialize RAG Chatbot
if GOOGLE_API_KEY:
    rag_bot = RAGChatbot(GOOGLE_API_KEY)
else:
    st.error("❌ GOOGLE_API_KEY not found in environment variables")
    st.stop()

# Sidebar for vectorstore management
with st.sidebar:
    st.header("📚 Fluentiq RAG Assistant")
    st.write("Manage your vector database and chat settings")
    
    # Vectorstore loading section
    st.subheader("🗂️ Vector Database")
    
    vectorstore_name = st.text_input(
        "Vectorstore Name", 
        value=st.session_state.vectorstore_name,
        help="Name of the FAISS vectorstore directory"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Load Vectorstore"):
            with st.spinner("Loading vectorstore..."):
                vectorstore, message = rag_bot.load_vectorstore(vectorstore_name)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.vectorstore_loaded = True
                    st.session_state.vectorstore_name = vectorstore_name
                    st.success("✅ " + message)
                else:
                    st.error("❌ " + message)
    
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Vectorstore status
    if st.session_state.vectorstore_loaded:
        st.success("✅ Vectorstore loaded successfully!")
        st.info("📄 RAG is active - I can answer questions about your documents!")
        
        # Test search functionality
        with st.expander("🔍 Test Search"):
            test_query = st.text_input("Test query:")
            if test_query and st.button("Search"):
                docs = rag_bot.retrieve_relevant_chunks(test_query, st.session_state.vectorstore, k=2)
                if docs:
                    for i, doc in enumerate(docs, 1):
                        st.text_area(f"Result {i}:", doc.page_content[:300] + "...", height=100)
                else:
                    st.write("No results found")
    else:
        st.warning("💡 Load a vectorstore to enable RAG!")
        st.info("Create vectorstore using 'create_vectordb.py' first")
    
    # Available vectorstores
    st.subheader("📂 Available Vectorstores")
    vectorstores = [d for d in os.listdir('.') if os.path.isdir(d) and any(f.endswith('.faiss') for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))]
    if vectorstores:
        for vs in vectorstores:
            if st.button(f"📁 {vs}", key=f"vs_{vs}"):
                st.session_state.vectorstore_name = vs
                st.rerun()
    else:
        st.write("No vectorstores found in current directory")

# Main interface
st.title("🗣️ Fluentiq - RAG-Powered AI Assistant")
st.markdown("""
**An intelligent document Q&A system using Retrieval-Augmented Generation (RAG)**
- 🧠 Powered by Google's Gemini AI and FAISS vector database
- 📚 Semantic search across your documents
- ⚡ Optimized for fast responses with pre-computed embeddings
""")

# Add demo info
st.info("""
**Demo Features**: This system can answer questions about uploaded documents using advanced AI techniques including:
semantic search, document chunking, and context-aware response generation.
""")

# Display current status
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.vectorstore_loaded:
        st.success("🎯 RAG: ACTIVE")
    else:
        st.warning("📄 RAG: INACTIVE")

with col2:
    st.info(f"📂 Current: {st.session_state.vectorstore_name}")

with col3:
    st.info(f"💬 Chat History: {len(st.session_state.chat_history)//2} exchanges")

# Display chat history
for i, (speaker, message) in enumerate(st.session_state.chat_history):
    with st.chat_message(speaker):
        st.markdown(message)

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the query with RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_container = st.empty()
            
            # Generate response
            bot_reply = rag_bot.generate_response(user_input, st.session_state.vectorstore)
            
            # Display the response
            response_container.markdown(bot_reply)
            
            # Add to chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", bot_reply))

# Project Information Footer
st.markdown("---")
st.markdown("### 🚀 **Project: RAG-Powered Document Q&A System**")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Technologies Used:**")
    st.markdown("""
    - LangChain (Document Processing)
    - FAISS (Vector Database)  
    - Google Gemini AI (LLM)
    - Streamlit (Web Interface)
    - Python (Backend Logic)
    """)

with col2:
    st.markdown("**📊 Technical Features:**")
    st.markdown("""
    - Semantic Search
    - Document Chunking
    - Embedding Generation
    - Context-Aware Responses
    - Real-time Chat Interface
    """)

with col3:
    st.markdown("**⚡ Optimizations:**")
    st.markdown("""
    - Pre-computed Vector DB
    - Efficient Chunk Retrieval
    - Modular Architecture
    - Fast Response Times
    - Scalable Design
    """)

# Footer with instructions
st.markdown("---")
with st.expander("💡 How to use RAG functionality"):
    st.markdown("""
    ### 🚀 Getting Started:
    
    **Step 1: Create Vector Database**
    ```bash
    python create_vectordb.py
    ```
    
    **Step 2: Load Vectorstore**
    - Enter vectorstore name in sidebar (e.g., 'faiss_vectorstore')
    - Click "📁 Load Vectorstore"
    
    **Step 3: Chat with RAG**
    - Ask questions about your documents
    - RAG will search relevant chunks automatically
    
    ### 📝 RAG Features:
    - **Document-based answers**: Prioritizes information from your documents
    - **Semantic search**: Finds most relevant content chunks
    - **Fallback to web search**: When document doesn't contain relevant info
    - **Source attribution**: Clear indication of information sources
    
    ### 🎯 Example Queries:
    - "What are the main points in my document?"
    - "Explain the methodology mentioned in the paper"
    - "Summarize the conclusions from my report"
    - "What does the document say about [specific topic]?"
    
    ### 📁 Vectorstore Management:
    - Vectorstores are saved as directories (e.g., `faiss_vectorstore/`)
    - Copy vectorstore directories to your project folder
    - Load different vectorstores for different document sets
    - Test search functionality using the sidebar tool
    """)

# Performance tips
with st.expander("⚡ Performance Tips"):
    st.markdown("""
    ### 🔧 Optimization Tips:
    
    **Vector Database Creation:**
    - Create vectorstore once, reuse multiple times
    - Chunk size: 1000 characters (optimal for most documents)
    - Overlap: 200 characters (maintains context)
    
    **RAG Configuration:**
    - Retrieve top 3 chunks by default (balance between context and speed)
    - Use semantic search for better relevance
    - Fallback to web search when needed
    
    **Git Workflow:**
    ```bash
    # After creating vectorstore
    git add faiss_vectorstore/
    git commit -m "Add vectorstore for RAG"
    git push
    ```
    
    **File Organization:**
    ```
    project/
    ├── create_vectordb.py      # Run once to create DB
    ├── rag_chatbot.py          # Main chatbot app
    ├── faiss_vectorstore/      # Generated vector DB
    ├── .env                    # API keys
    └── requirements.txt        # Dependencies
    ```
    """)

# Display current RAG status at bottom
if st.session_state.vectorstore_loaded:
    st.success("🎯 RAG is ACTIVE - I can answer questions using your vector database!")
    st.info(f"💡 Using vectorstore: '{st.session_state.vectorstore_name}'")
else:
    st.warning("📄 No vectorstore loaded - Load a vectorstore to enable RAG functionality!")
    st.info("🌐 I can still help with general questions and web search.")