import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import tempfile
import requests
from bs4 import BeautifulSoup
import io
from dotenv import load_dotenv

load_dotenv()  # Loads .env file

api_key = os.getenv("GOOGLE_API_KEY")


# Streamlit Config
st.set_page_config(page_title="Fluentiq RAG", page_icon="🗣️", layout="wide")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📚 Fluentiq RAG Assistant")
    st.write("Upload documents for RAG-powered conversations!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=['txt', 'pdf', 'docx', 'md'],
        help="Upload text files, PDFs, Word docs, or Markdown files"
    )
    
    if st.button("🔄 Clear Chat & Documents"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.document_loaded = False
        st.rerun()
    
    if st.session_state.document_loaded:
        st.success("✅ Document loaded successfully!")
        st.info("📄 RAG is active - I can answer questions about your document!")
    else:
        st.info("💡 Upload a document to enable RAG!")

# Main interface
st.title("🗣️ Fluentiq - RAG-Powered AI Assistant")
st.write("Ask me anything! I'm here to help with information and answer your questions.")

def extract_text_from_file(uploaded_file):
    """Extract text from various file types"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "text/markdown":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except ImportError:
                st.error("PyPDF2 not installed. Please install it to read PDF files.")
                return None
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                import python_docx
                doc = python_docx.Document(io.BytesIO(uploaded_file.read()))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                st.error("python-docx not installed. Please install it to read Word documents.")
                return None
        else:
            # Try to read as text
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def create_vectorstore_from_text(text):
    """Create vectorstore from text content"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        # Create documents
        docs = [Document(page_content=text)]
        split_docs = text_splitter.split_documents(docs)
        
        if not split_docs:
            return None
            
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def search_web(query):
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

def retrieve_relevant_chunks(query, vectorstore, k=3):
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

@st.cache_resource
def get_chat_model():
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
        return model.start_chat()
    except Exception as e:
        st.error(f"Error initializing chat model: {e}")
        return None

# Process uploaded file
if uploaded_file and not st.session_state.document_loaded:
    with st.spinner("Processing your document for RAG..."):
        text_content = extract_text_from_file(uploaded_file)
        
        if text_content:
            vectorstore = create_vectorstore_from_text(text_content)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.document_loaded = True
                st.success(f"✅ Successfully processed '{uploaded_file.name}' for RAG!")
                st.info("🚀 RAG is now active! Ask questions about your document.")
            else:
                st.error("Failed to create vector store from the document.")
        else:
            st.error("Could not extract text from the uploaded file.")

# Initialize chat model
chat = get_chat_model()
if chat is None:
    st.error("❌ Could not initialize chat model. Please check your API key.")
    st.stop()

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
            
            try:
                context_info = ""
                
                # RAG: Retrieve relevant chunks from uploaded document
                if st.session_state.vectorstore:
                    with st.spinner("Retrieving relevant information..."):
                        retrieved_docs = retrieve_relevant_chunks(user_input, st.session_state.vectorstore, k=3)
                        
                        if retrieved_docs:
                            doc_contexts = []
                            for i, doc in enumerate(retrieved_docs, 1):
                                doc_contexts.append(f"Document Chunk {i}:\n{doc.page_content}")
                            
                            context_info = "Retrieved Context from Your Document:\n" + "\n\n".join(doc_contexts) + "\n\n"
                
                # Fallback to web search if no document context available
                if not context_info:
                    with st.spinner("Searching the web..."):
                        web_results = search_web(user_input)
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
                response = chat.send_message(prompt)
                bot_reply = response.text
                
                # Display the response
                response_container.markdown(bot_reply)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", bot_reply))
                
            except Exception as e:
                error_message = f"I'm sorry, I encountered an error: {str(e)[:100]}... Please try again!"
                response_container.markdown(error_message)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", error_message))

# Footer with RAG tips
st.markdown("---")
with st.expander("💡 Tips for RAG-powered conversations"):
    st.markdown("""
    **For document-based RAG:**
    - Upload relevant documents (PDF, Word, text files, Markdown)
    - Ask specific questions about the document content
    - Reference particular topics, concepts, or sections
    - Ask for summaries or explanations of document content
    
    **RAG Features:**
    - Semantic search finds the most relevant document sections
    - Multiple chunks are retrieved for comprehensive answers
    - Clear indication when using document vs. general knowledge
    
    **Example RAG queries:**
    - "What are the main points discussed in my document?"
    - "Explain the methodology mentioned in the uploaded paper"
    - "Summarize the conclusions from my report"
    - "What does the document say about [specific topic]?"
    """)

# Display current RAG status
if st.session_state.document_loaded:
    st.success("🎯 RAG is ACTIVE - I can answer questions using your uploaded document!")
    st.info("💡 I'll search your document first, then use general knowledge if needed.")
else:
    st.warning("📄 No document uploaded - Upload a document to enable RAG functionality!")
    st.info("🌐 I can still help with general questions and web search.")