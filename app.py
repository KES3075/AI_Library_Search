"""
NU eLibrary RAG Chatbot
A search interface for National University eLibrary with AI-powered recommendations
Enhanced with 65535 max tokens for comprehensive responses
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
        """Initialize RAG system with ChromaDB and Gemini with extended token limit"""
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
        
        # Initialize Gemini with standard configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("⚠️ GOOGLE_API_KEY not found. Please set it in .env file")
            st.stop()

        genai.configure(api_key=api_key)

        # Configure generation with standard settings
        self.generation_config = genai.GenerationConfig(
            max_output_tokens=8000,  # Limit for concise summaries
            temperature=0,
        )
        
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=self.generation_config
        )
        
    
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
    
    def generate_summary(self, book_info, query, chunk_type='keyword_summary'):
        """Generate concise AI summary for different types of chunks based on query"""
        # Customize prompt based on chunk type
        if chunk_type == 'keyword_summary':
            prompt = f"""You are an expert academic librarian at National University eLibrary.

KEYWORD COLLECTION DATA:
{book_info}

USER'S RESEARCH QUERY: {query}

TASK: Provide a concise one-paragraph summary explaining how this keyword collection relates to the user's research query. Focus on the key resources, main themes, and practical value for researchers.

WRITING GUIDELINES:
- Keep response to one paragraph (3-5 sentences, 100-150 words)
- Highlight the most relevant books, authors, and research themes
- Explain the collection's value for the specific query
- Use specific details from the data when available
- Maintain professional academic tone

Begin your response directly with the summary."""

        elif chunk_type == 'author_summary':
            prompt = f"""You are an expert academic librarian at National University eLibrary.

AUTHOR PROFILE DATA:
{book_info}

USER'S RESEARCH QUERY: {query}

TASK: Provide a concise one-paragraph summary of this author's scholarly profile and how their work relates to the user's research query. Focus on their key expertise areas, major publications, and research value.

WRITING GUIDELINES:
- Keep response to one paragraph (3-5 sentences, 100-150 words)
- Highlight the author's main research focus and relevant publications
- Explain how their expertise connects to the specific query
- Use specific details from the data when available
- Maintain professional academic tone

Begin your response directly with the author summary."""

        else:  # book_detail or default
            prompt = f"""You are an expert academic librarian at National University eLibrary.

BOOK INFORMATION:
{book_info}

USER'S RESEARCH QUERY: {query}

TASK: Provide a concise one-paragraph summary of this book and its relevance to the user's research query. Focus on the book's main content, key contributions, and research value.

WRITING GUIDELINES:
- Keep response to one paragraph (3-5 sentences, 100-150 words)
- Describe the book's main topic and key content
- Explain how it relates to the specific query
- Highlight the book's value for research or study
- Use specific details from the data when available
- Maintain professional academic tone

Begin your response directly with the book summary."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback when API fails - extract detailed information from book_info
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return self._generate_detailed_fallback_summary(book_info, query, chunk_type)
            return self._generate_basic_fallback_summary(book_info, query, chunk_type)

    def _generate_detailed_fallback_summary(self, book_info, query, chunk_type):
        """Generate concise fallback summary by parsing book_info when API quota is exceeded"""
        if not isinstance(book_info, str):
            return f"Academic content available for '{query}' in the NU eLibrary collection. (API temporarily unavailable)"

        lines = book_info.split('\n')
        extracted_info = {}

        # Extract information from the embedded text
        for line in lines:
            line = line.strip()
            if line.startswith('Author:') and 'Author:' not in extracted_info:
                author = line.replace('Author:', '').strip()
                # Clean up author names
                author = author.replace('Dr ', '').replace('Professor ', '').replace('faculty ', '')
                if author and author != 'Unknown':
                    extracted_info['author'] = author

            elif line.startswith('Title:') and 'Title:' not in extracted_info:
                title = line.replace('Title:', '').strip()
                if title and title != 'Unknown Title':
                    extracted_info['title'] = title

            elif line.startswith('Keyword:') and 'Keyword:' not in extracted_info:
                keyword = line.replace('Keyword:', '').strip()
                if keyword and keyword != 'Unknown':
                    extracted_info['keyword'] = keyword

            elif line.startswith('College:') and 'College:' not in extracted_info:
                college = line.replace('College:', '').strip()
                if college and college != 'Unknown':
                    extracted_info['college'] = college

            elif line.startswith('Availability:') and 'Availability:' not in extracted_info:
                availability = line.replace('Availability:', '').strip()
                extracted_info['availability'] = availability

            elif line.startswith('Publication:') and 'Publication:' not in extracted_info:
                publication = line.replace('Publication:', '').strip()
                extracted_info['publication'] = publication

            elif line.startswith('Description:') and 'Description:' not in extracted_info:
                description = line.replace('Description:', '').strip()
                if len(description) > 50:  # Only if substantial description
                    extracted_info['description'] = description[:200] + '...' if len(description) > 200 else description

        # Generate concise one-paragraph summary based on chunk type
        if chunk_type == 'keyword_summary':
            keyword = extracted_info.get('keyword', query)
            return f"This collection covers '{keyword}' with resources relevant to '{query}', providing valuable academic materials for research and study in this discipline."

        elif chunk_type == 'author_summary':
            author = extracted_info.get('author', 'NU faculty')
            keyword = extracted_info.get('keyword', 'academic subjects')
            return f"{author} is a researcher specializing in {keyword}, with publications relevant to '{query}' that contribute to academic discourse in this field."

        else:  # book_detail
            title = extracted_info.get('title', 'This publication')
            author = extracted_info.get('author', 'the author')
            return f"'{title}' by {author} is a valuable resource for '{query}' research, providing important insights and information for academic study."

    def _generate_basic_fallback_summary(self, book_info, query, chunk_type):
        """Generate basic fallback summary for other API errors"""
        if chunk_type == 'keyword_summary':
            return f"Academic collection covering '{query}' with multiple scholarly resources available in the NU eLibrary."
        elif chunk_type == 'author_summary':
            return f"Faculty research profile with publications relevant to '{query}' available for reference."
        else:
            return f"Scholarly publication relevant to '{query}' available in the NU eLibrary collection."

    def generate_recommendations(self, query, search_results):
        """Generate comprehensive recommendations using Gemini with extended token limit"""
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

Based on the above publications, provide a concise 4-5 sentence analysis that covers:

1. **Key Findings**: Summarize the main themes and most relevant publications found for this query
2. **Resource Value**: Explain the academic significance and practical applications of these materials
3. **Research Guidance**: Provide brief recommendations on how to use these resources effectively

Keep the response conversational, helpful, and focused on the user's specific research needs. Limit to exactly 4-5 sentences total."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return self._generate_detailed_recommendation_fallback(query, search_results)
            return self._generate_basic_recommendation_fallback(query, search_results)

    def _generate_detailed_recommendation_fallback(self, query, search_results):
        """Generate concise recommendation fallback when API quota exceeded"""
        num_results = len(search_results['documents'][0])

        # Analyze the results to extract key information
        keywords_found = set()
        authors_found = set()
        chunk_types = {'keyword_summary': 0, 'author_summary': 0, 'book_detail': 0}

        for metadata in search_results['metadatas'][0]:
            chunk_type = metadata.get('chunk_type', 'book_detail')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            if metadata.get('keyword'):
                keywords_found.add(metadata['keyword'])
            if metadata.get('author'):
                authors_found.add(metadata['author'])

        # Create concise 4-5 sentence summary
        sentences = []

        # Key findings
        sentences.append(f"I found {num_results} relevant resources for '{query}' in the NU eLibrary collection.")

        # Main themes and resource types
        resource_types = []
        if chunk_types.get('keyword_summary', 0) > 0:
            resource_types.append(f"{chunk_types['keyword_summary']} subject overview(s)")
        if chunk_types.get('author_summary', 0) > 0:
            resource_types.append(f"{chunk_types['author_summary']} author profile(s)")
        if chunk_types.get('book_detail', 0) > 0:
            resource_types.append(f"{chunk_types['book_detail']} specific book(s)")

        if resource_types:
            sentences.append(f"The results include {', '.join(resource_types)} that directly relate to your research topic.")

        # Academic value
        if keywords_found:
            keyword_list = list(keywords_found)[:2]
            sentences.append(f"Key academic areas covered include {', '.join(keyword_list)}, providing valuable insights for your research.")

        # Research guidance
        sentences.append("These materials offer comprehensive coverage of your topic with diverse perspectives from NU faculty and researchers.")

        if len(sentences) > 5:
            sentences = sentences[:5]

        return ' '.join(sentences)

    def _generate_basic_recommendation_fallback(self, query, search_results):
        """Generate concise basic recommendation fallback for other API errors"""
        num_results = len(search_results['documents'][0])
        return f"I found {num_results} relevant resources for '{query}' in the NU eLibrary collection. These materials provide valuable insights for your research and are available for reference."


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

    # Generate AI summary with chunk type awareness first
    try:
        with st.spinner(f"Generating summary for #{rank}..."):
            summary = rag_system.generate_summary(document, query, chunk_type)
    except Exception as e:
        # Fallback summary based on chunk type
        if chunk_type == 'keyword_summary':
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"This collection covers '{keyword}' with resources relevant to '{query}', providing valuable academic materials for research and study."
        elif chunk_type == 'author_summary':
            author = metadata.get('author', 'NU Faculty')
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"{author} is a researcher specializing in {keyword}, with publications relevant to '{query}' that contribute to academic discourse."
        else:
            author = metadata.get('author', 'NU Faculty')
            keyword = metadata.get('keyword', 'Academic Subject')
            summary = f"A library book by {author} in '{keyword}', providing valuable insights for '{query}' research."

    # Create complete book card with all content in a single HTML block
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
        <div class="book-summary">
            <strong>🤖 AI Summary:</strong><br>
            {summary}
        </div>
        <div style="margin-top: 0.8rem;">
            <a href="{button_url}" target="_blank" style="background-color: #c6982c; color: white; padding: 0.5rem 1rem; border-radius: 5px; text-decoration: none; font-weight: bold; display: inline-block;">
                {button_text}
            </a>
        </div>
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


def main():
    """Main application"""
    # Header
    st.markdown("""
        <div class="header-banner">
            <h1>🧠 NU eLibrary Intelligent Search</h1>
            <p style="margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.9;">Keyword-Based AI-Powered Book Discovery with Concise Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("🔧 Initializing AI system..."):
            st.session_state.rag_system = LibraryRAG()
    
    rag_system = st.session_state.rag_system
    
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
                with st.spinner("Generating analysis..."):
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
                st.info("🚀 **Concise Analysis:** This version provides focused, one-paragraph AI-generated summaries for quick reference.")

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
            <p style="color: #c6982c; font-weight: bold; margin-top: 1rem;">
                ⚡ Enhanced with AI-powered concise analysis for quick reference
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