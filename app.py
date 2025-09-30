"""
NU eLibrary RAG Chatbot
A search interface for National University eLibrary with AI-powered recommendations
"""
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="NU eLibrary Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #c6982c;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #b08825;
    }
    .book-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #c6982c;
    }
    .book-title {
        color: #192f59;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .book-author {
        color: #545454;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .book-summary {
        color: #666;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .book-links {
        margin-top: 1rem;
    }
    .book-links a {
        background-color: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        text-decoration: none;
        color: #192f59;
        margin-right: 0.5rem;
        display: inline-block;
        margin-top: 0.3rem;
    }
    .book-links a:hover {
        background-color: #e0e0e0;
    }
    .header-banner {
        background: linear-gradient(135deg, #192f59 0%, #c6982c 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .relevance-score {
        background-color: #c6982c;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class LibraryRAG:
    def __init__(self):
        """Initialize RAG system with ChromaDB and Gemini"""
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.collection = self.client.get_collection(
                name="nu_library_books",
                embedding_function=self.embedding_function
            )
        except:
            st.error("⚠️ Database not found. Please run data_ingestion.py first!")
            st.stop()
        
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("⚠️ GOOGLE_API_KEY not found. Please set it in .env file")
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def search_books(self, query, n_results=5):
        """Search for books using semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
    
    def generate_summary(self, book_info, query):
        """Generate AI summary for a book based on query"""
        prompt = f"""You are a knowledgeable librarian assistant for National University eLibrary.

Given this publication/book information:
{book_info}

User Query: {query}

Provide a concise, informative summary (2-3 sentences) that:
1. Highlights why this publication is relevant to the user's query
2. Mentions the author's expertise area if available
3. Provides context about the publication type

Keep it professional and academic in tone."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback when API fails - extract author from the text
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                # Extract author name from book_info string if possible
                if isinstance(book_info, str) and 'Author:' in book_info:
                    try:
                        author_line = [line for line in book_info.split('\n') if line.startswith('Author:')][0]
                        author = author_line.replace('Author:', '').strip()
                        return f"A faculty publication by {author}. This work is available for reference in the NU eLibrary collection and is relevant to your search on {query}."
                    except:
                        pass
                return f"A faculty publication available in the NU eLibrary collection, relevant to your search on '{query}'. (AI summary unavailable due to API quota limits - please wait a moment and refresh)"
            return f"A faculty publication available in the NU eLibrary collection."
    
    def generate_recommendations(self, query, search_results):
        """Generate comprehensive recommendations using Gemini"""
        # Prepare context from search results
        context = "Retrieved Publications:\n\n"
        for i, (doc, meta, dist) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            context += f"{i+1}. {doc}\n"
            context += f"   Relevance Score: {1 - dist:.2f}\n\n"
        
        prompt = f"""You are an AI librarian for National University eLibrary (https://elibrary.nu.edu.om/).

User Query: {query}

{context}

Based on the above publications, provide:
1. A brief introduction addressing the user's query
2. Explain how these publications relate to their search
3. Mention any notable authors or research areas

Keep it conversational, helpful, and academic. Limit to 4-5 sentences."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return f"I found {len(search_results['documents'][0])} relevant publications for '{query}' in our collection. These faculty publications span various research areas and are available for reference. (AI analysis temporarily unavailable due to API quota - summaries will return shortly)"
            return "I found several relevant publications in our collection that might interest you."


def display_book_card(book_data, rank, query, rag_system):
    """Display a single book card with AI-generated summary"""
    metadata = book_data['metadata']
    document = book_data['document']
    distance = book_data['distance']
    
    # Calculate relevance (distance is between 0 and 2, lower is better)
    # Convert to percentage where lower distance = higher relevance
    relevance = max(0, min(100, (1 - distance) * 100))
    
    # Extract author from metadata or document
    author = metadata.get('author', 'NU Faculty')
    if author == 'NU Faculty' and 'Author:' in document:
        try:
            author_line = [line for line in document.split('\n') if line.startswith('Author:')][0]
            author = author_line.replace('Author:', '').strip()
        except:
            pass
    
    # Create book card with link to original publication
    detail_url = metadata.get('detail_url', metadata.get('url', 'https://elibrary.nu.edu.om/'))
    
    st.markdown(f"""
    <div class="book-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div class="book-title">
                    <a href="{detail_url}" target="_blank" style="color: #192f59; text-decoration: none;">
                        #{rank}. {metadata.get('title', 'Faculty Publication')} 🔗
                    </a>
                </div>
                <div class="book-author">
                    👤 {author}
                </div>
            </div>
            <span class="relevance-score">
                {relevance:.0f}% Match
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Generate AI summary with error handling
    try:
        with st.spinner(f"Generating summary for #{rank}..."):
            summary = rag_system.generate_summary(document, query)
    except Exception as e:
        # Fallback summary if AI fails
        college = metadata.get('college', 'National University')
        summary = f"A faculty publication by {author} from {college}, available for reference in the NU eLibrary collection."
    
    st.markdown(f"""
        <div class="book-summary">
            <strong>📝 AI Summary:</strong><br>
            {summary}
        </div>
        <div style="margin-top: 0.8rem;">
            <a href="{detail_url}" target="_blank" style="background-color: #c6982c; color: white; padding: 0.5rem 1rem; border-radius: 5px; text-decoration: none; font-weight: bold; display: inline-block;">
                🔍 View Full Details on NU eLibrary
            </a>
        </div>
    """, unsafe_allow_html=True)

    # Display additional info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if metadata.get('college'):
            st.markdown(f"**🏛️ College:** {metadata['college']}")
        if metadata.get('item_type'):
            st.markdown(f"**📚 Type:** {metadata['item_type']}")
        if metadata.get('availability'):
            st.markdown(f"**📍 Availability:** {metadata['availability']}")
    
    with col2:
        # Research profile links
        st.markdown('<div class="book-links">', unsafe_allow_html=True)
        if metadata.get('google_scholar'):
            st.markdown(f"[🎓 Google Scholar]({metadata['google_scholar']})")
        if metadata.get('research_gate'):
            st.markdown(f"[🔬 ResearchGate]({metadata['research_gate']})")
        if metadata.get('scopus'):
            st.markdown(f"[📊 Scopus]({metadata['scopus']})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main application"""
    # Header
    st.markdown("""
        <div class="header-banner">
            <h1>📚 NU eLibrary Search</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("🔧 Initializing AI system..."):
            st.session_state.rag_system = LibraryRAG()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar removed for cleaner interface
    
    # Main search interface
    st.markdown("### 🔎 Search Publications")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., mechanical engineering, faculty publications, Dr Ahmed...",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Perform search
    if query and (search_button or query):
        st.markdown("---")
        
        with st.spinner("🔍 Searching through publications..."):
            # Search for books
            results = rag_system.search_books(query, n_results=5)
            
            if results['documents'][0]:
                # Generate AI introduction
                st.markdown("### 🤖 AI Analysis")
                with st.spinner("Analyzing results..."):
                    intro = rag_system.generate_recommendations(query, results)
                
                st.info(intro)
                
                st.markdown("---")
                st.markdown("### 📚 Top 5 Recommended Publications")
                
                # Display results
                for i in range(len(results['documents'][0])):
                    book_data = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }
                    
                    display_book_card(book_data, i + 1, query, rag_system)
                
                # Footer
                st.markdown("---")
                st.success(f"✅ Found {len(results['documents'][0])} relevant publications for your query")
                
            else:
                st.warning("😕 No publications found matching your query. Try different keywords!")
    
    elif not query:
        # Welcome message
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h3 style="color: #192f59;">👋 Welcome to NU eLibrary Search!</h3>
            <p style="color: #666; font-size: 1.1rem; margin-top: 1rem;">
                Enter a search query above to discover faculty publications and research.
            </p>
            <p style="color: #666;">
                Our AI-powered system will find the most relevant publications and 
                provide you with intelligent summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9rem;">
            <strong>National University eLibrary Search</strong><br>
            Copyright © 2025, National University Libraries<br>
            Azaiba, Bousher, Muscat, Sultanate of Oman<br>
            <a href="https://elibrary.nu.edu.om/" target="_blank">Visit Official Website</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
