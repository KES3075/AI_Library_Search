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
    page_title="NU eLibrary Intelligent Search",
    page_icon="🧠",
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
    
    def search_books(self, query, n_results=10, prioritize_publications=True):
        """Search for books using semantic search with hierarchical chunking awareness"""
        # Get more results initially to allow for filtering/prioritization
        initial_results = n_results * 3

        results = self.collection.query(
            query_texts=[query],
            n_results=initial_results,
            include=['documents', 'metadatas', 'distances']
        )

        if not results['documents'][0]:
            return results

        # Process results based on hierarchical structure
        processed_results = self._process_hierarchical_results(
            results, query, n_results, prioritize_publications
        )

        return processed_results

    def _process_hierarchical_results(self, results, query, max_results, prioritize_publications):
        """Process and prioritize hierarchical search results"""
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Group results by chunk type and level
        level_1_chunks = []  # Keyword summaries
        level_2_chunks = []  # Author summaries
        level_3_chunks = []  # Individual books

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunk_level = meta.get('chunk_level', 3)
            chunk_type = meta.get('chunk_type', 'publication_detail')

            result_item = {
                'document': doc,
                'metadata': meta,
                'distance': dist,
                'chunk_level': chunk_level,
                'chunk_type': chunk_type
            }

            if chunk_level == 1:
                level_1_chunks.append(result_item)
            elif chunk_level == 2:
                level_2_chunks.append(result_item)
            else:
                level_3_chunks.append(result_item)

        # Prioritize results based on strategy
        final_results = []

        if prioritize_publications:
            # Strategy: Show some keyword context, then author info, then books
            # Take top 1 keyword summary if available
            if level_1_chunks:
                final_results.extend(sorted(level_1_chunks, key=lambda x: x['distance'])[:1])

            # Take top 2 author summaries if available
            if level_2_chunks:
                final_results.extend(sorted(level_2_chunks, key=lambda x: x['distance'])[:2])

            # Fill remaining slots with books
            remaining_slots = max_results - len(final_results)
            if remaining_slots > 0 and level_3_chunks:
                final_results.extend(sorted(level_3_chunks, key=lambda x: x['distance'])[:remaining_slots])
        else:
            # Alternative strategy: Pure relevance-based ranking
            all_chunks = level_1_chunks + level_2_chunks + level_3_chunks
            final_results = sorted(all_chunks, key=lambda x: x['distance'])[:max_results]

        # Reformat to match original structure
        processed_results = {
            'documents': [[r['document'] for r in final_results]],
            'metadatas': [[r['metadata'] for r in final_results]],
            'distances': [[r['distance'] for r in final_results]]
        }

        return processed_results
    
    def generate_summary(self, book_info, query, chunk_type='publication_detail'):
        """Generate AI summary for different types of chunks based on query"""
        # Customize prompt based on chunk type
        if chunk_type == 'keyword_summary':
            prompt = f"""You are a knowledgeable librarian assistant for National University eLibrary.

Given this keyword overview information:
{book_info}

User Query: {query}

Provide a concise summary (2-3 sentences) that:
1. Explains how this academic keyword relates to the user's query
2. Highlights key books and research areas covered
3. Mentions the scope of available resources in this subject area

Keep it professional and academic in tone."""

        elif chunk_type == 'author_summary':
            prompt = f"""You are a knowledgeable librarian assistant for National University eLibrary.

Given this author profile information:
{book_info}

User Query: {query}

Provide a concise summary (2-3 sentences) that:
1. Describes the author's research focus and how it relates to the query
2. Notes their publication activity and research networks
3. Highlights their expertise areas and available works

Keep it professional and academic in tone."""

        else:  # book_detail or default
            prompt = f"""You are a knowledgeable librarian assistant for National University eLibrary.

Given this book information:
{book_info}

User Query: {query}

Provide a concise, informative summary (2-3 sentences) that:
1. Highlights why this book is relevant to the user's query
2. Mentions the author's expertise area if available
3. Provides context about the book type and availability

Keep it professional and academic in tone."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback when API fails - customize based on chunk type
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                if chunk_type == 'keyword_summary':
                    return f"A comprehensive overview of books and resources in this academic subject area, relevant to your search on '{query}'. (AI summary unavailable due to API quota limits - please wait a moment and refresh)"
                elif chunk_type == 'author_summary':
                    return f"A faculty researcher's profile with multiple books, relevant to your search on '{query}'. (AI summary unavailable due to API quota limits - please wait a moment and refresh)"
                else:
                    # Extract author name from book_info string if possible
                    if isinstance(book_info, str) and 'Author:' in book_info:
                        try:
                            author_line = [line for line in book_info.split('\n') if line.startswith('Author:')][0]
                            author = author_line.replace('Author:', '').strip()
                            return f"A book by {author}. This work is available for reference in the NU eLibrary collection and is relevant to your search on {query}."
                        except:
                            pass
                    return f"A book available in the NU eLibrary collection, relevant to your search on '{query}'. (AI summary unavailable due to API quota limits - please wait a moment and refresh)"
            return f"Academic content available in the NU eLibrary collection."
    
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
    """Display a single book card with AI-generated summary, adapted for hierarchical chunks"""
    metadata = book_data['metadata']
    document = book_data['document']
    distance = book_data['distance']
    chunk_type = metadata.get('chunk_type', 'publication_detail')
    chunk_level = metadata.get('chunk_level', 3)

    # Determine card styling and content based on chunk type
    if chunk_type == 'keyword_summary':
        card_icon = "🏷️"
        card_title = f"Keyword Overview: {metadata.get('keyword', 'Academic Subject')}"
        card_subtitle = "📊 Book Collection Overview"
        button_text = "🔍 Explore Books in this Subject"
        keyword_search = metadata.get('keyword', '').replace(' ', '+')
        button_url = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl?q={keyword_search}"
    elif chunk_type == 'author_summary':
        card_icon = "👤"
        card_title = f"Author Profile: {metadata.get('author', 'NU Faculty')}"
        card_subtitle = f"📚 {metadata.get('book_count', 0)} Books"
        button_text = "🔍 View Author Books"
        author_name = metadata.get('author', '').replace(' ', '+')
        button_url = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl?q=au:%22{author_name}%22"
    else:  # book_detail
        card_icon = "📖"
        card_title = metadata.get('title', 'Library Book')

        # Extract author from metadata or document
        author = metadata.get('author', 'NU Faculty')
        if author == 'NU Faculty' and 'Author:' in document:
            try:
                author_line = [line for line in document.split('\n') if line.startswith('Author:')][0]
                author = author_line.replace('Author:', '').strip()
            except:
                pass
        card_subtitle = f"👤 {author}"
        button_text = "🔍 View Full Details on NU eLibrary"
        button_url = metadata.get('link', 'https://elibrary.nu.edu.om/')

    # Create book card with appropriate styling
    st.markdown(f"""
    <div class="book-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div class="book-title">
                    <a href="{button_url}" target="_blank" style="color: #192f59; text-decoration: none;">
                        #{rank}. {card_title} 🔗
                    </a>
                </div>
                <div class="book-author">
                    {card_subtitle}
                </div>
                <div style="margin-top: 0.3rem; font-size: 0.85rem; color: #666;">
                    📊 Chunk Level {chunk_level} • {chunk_type.replace('_', ' ').title()}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Generate AI summary with chunk type awareness
    try:
        with st.spinner(f"Generating summary for #{rank}..."):
            summary = rag_system.generate_summary(document, query, chunk_type)
    except Exception as e:
        # Fallback summary based on chunk type
        if chunk_type == 'keyword_summary':
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"A comprehensive overview of books and resources in '{keyword}', relevant to your search on '{query}'."
        elif chunk_type == 'author_summary':
            author = metadata.get('author', 'NU Faculty')
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"Profile of {author} with {metadata.get('book_count', 0)} books in '{keyword}', relevant to your search on '{query}'."
        else:
            author = metadata.get('author', 'NU Faculty')
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"A library book by {author} in '{keyword}', available for reference in the NU eLibrary collection."

    st.markdown(f"""
        <div class="book-summary">
            <strong>🤖 AI Summary:</strong><br>
            {summary}
        </div>
        <div style="margin-top: 0.8rem;">
            <a href="{button_url}" target="_blank" style="background-color: #c6982c; color: white; padding: 0.5rem 1rem; border-radius: 5px; text-decoration: none; font-weight: bold; display: inline-block;">
                {button_text}
            </a>
        </div>
    """, unsafe_allow_html=True)

    # Display additional info based on chunk type
    if chunk_type == 'keyword_summary':
        col1, col2 = st.columns([2, 1])
        with col1:
            if metadata.get('total_books'):
                st.markdown(f"**📚 Total Books:** {metadata['total_books']}")
            if metadata.get('total_authors'):
                st.markdown(f"**👥 Total Authors:** {metadata['total_authors']}")

        with col2:
            if metadata.get('college_stats'):
                st.markdown("**🏛️ Colleges:**")
                college_stats = eval(metadata['college_stats'])
                for college, count in college_stats.items():
                    st.markdown(f"  • {college}: {count}")

    elif chunk_type == 'author_summary':
        col1, col2 = st.columns([2, 1])
        with col1:
            if metadata.get('keyword'):
                st.markdown(f"**🏷️ Keyword:** {metadata['keyword']}")
            if metadata.get('book_count'):
                st.markdown(f"**📚 Books:** {metadata['book_count']}")
            if metadata.get('has_research_links') == 'True':
                st.markdown("**🔗 Research Networks:** Available")

        with col2:
            if metadata.get('colleges'):
                st.markdown("**🏛️ Colleges:**")
                for college in eval(metadata['colleges']):
                    st.markdown(f"  • {college}")

    else:  # book_detail
        col1, col2 = st.columns([2, 1])

        with col1:
            if metadata.get('keyword'):
                st.markdown(f"**🏷️ Keyword:** {metadata['keyword']}")
            if metadata.get('description'):
                # Extract availability from description
                description = metadata['description']
                if 'Not for loan' in description or 'Reference only' in description:
                    st.markdown("**📍 Availability:** Reference only")
                elif 'Available for loan' in description:
                    st.markdown("**📍 Availability:** Available for loan")
                elif 'Checked out' in description:
                    st.markdown("**📍 Availability:** Currently checked out")
                elif 'In transit' in description:
                    st.markdown("**📍 Availability:** In transit")

        with col2:
            # Extract college info from description
            description = metadata.get('description', '')
            if 'College of Engineering' in description:
                st.markdown("**🏛️ College:** College of Engineering")
            elif 'College of Medicine and Health Sciences' in description:
                st.markdown("**🏛️ College:** College of Medicine and Health Sciences")
            elif 'College of Pharmacy' in description:
                st.markdown("**🏛️ College:** College of Pharmacy")
            elif 'International Maritime College Oman' in description:
                st.markdown("**🏛️ College:** International Maritime College Oman")

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main application"""
    # Header
    st.markdown("""
        <div class="header-banner">
            <h1>🧠 NU eLibrary Intelligent Search</h1>
            <p style="margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.9;">Keyword-Based AI-Powered Book Discovery</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("🔧 Initializing AI system..."):
            st.session_state.rag_system = LibraryRAG()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar removed for cleaner interface
    
    # Main search interface
    st.markdown("### 🔎 Search Books")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., research methodology, qualitative research, machine learning, data analysis...",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Perform search
    if query and (search_button or query):
        st.markdown("---")
        
        with st.spinner("🔍 Searching through hierarchical knowledge base..."):
            # Search for books with hierarchical chunking
            results = rag_system.search_books(query, n_results=5)  # Limited to maximum 5 results

            if results['documents'][0]:
                # Generate AI introduction
                st.markdown("### 🤖 AI Analysis")
                with st.spinner("Analyzing hierarchical results..."):
                    intro = rag_system.generate_recommendations(query, results)

                st.info(intro)

                st.markdown("---")
                st.markdown("### 📚 Intelligent Results (Keyword-Based Search)")

                # Count chunk types for summary
                chunk_types = {}
                for metadata in results['metadatas'][0]:
                    chunk_type = metadata.get('chunk_type', 'publication_detail')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                # Display chunk type summary
                summary_parts = []
                if chunk_types.get('keyword_summary'):
                    summary_parts.append(f"🏷️ {chunk_types['keyword_summary']} Keyword Overview")
                if chunk_types.get('author_summary'):
                    summary_parts.append(f"👤 {chunk_types['author_summary']} Author Profile")
                if chunk_types.get('book_detail'):
                    summary_parts.append(f"📖 {chunk_types['book_detail']} Book")

                if summary_parts:
                    st.markdown(f"**Result Mix:** {' • '.join(summary_parts)}")
                    st.markdown("---")

                # Display results
                for i in range(len(results['documents'][0])):
                    book_data = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }

                    display_book_card(book_data, i + 1, query, rag_system)

                # Footer with hierarchical stats
                st.markdown("---")
                total_results = len(results['documents'][0])
                keyword_overviews = chunk_types.get('keyword_summary', 0)
                author_profiles = chunk_types.get('author_summary', 0)
                books = chunk_types.get('book_detail', 0)

                st.success(f"✅ Found {total_results} intelligently ranked results for your query")
                st.info("💡 **Hierarchical Search Benefits:** Keyword overviews provide subject context, author profiles show research focus, and individual books offer specific details.")

            else:
                st.warning("😕 No books found matching your query. Try different keywords!")
    
    elif not query:
        # Welcome message
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h3 style="color: #192f59;">👋 Welcome to NU eLibrary Intelligent Search!</h3>
            <p style="color: #666; font-size: 1.1rem; margin-top: 1rem;">
                Discover library books and resources with our AI-powered keyword-based search system.
            </p>
            <p style="color: #666;">
                Our intelligent system provides multi-level results: keyword overviews for subject context,
                author profiles for research focus, and individual books for specific details.
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
