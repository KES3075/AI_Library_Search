# 📚 NU eLibrary RAG Chatbot

AI-powered search and recommendation system for National University eLibrary faculty publications using RAG (Retrieval Augmented Generation).

## 🌟 Features

- **🤖 AI-Powered Search**: Semantic search using ChromaDB embeddings
- **✨ Smart Recommendations**: Top 5 book recommendations with AI-generated summaries
- **🎯 Accurate Results**: Uses Gemini 2.5 Flash for intelligent summarization
- **🔗 Research Profiles**: Direct links to Google Scholar, ResearchGate, and Scopus
- **📊 Real-time Stats**: Database statistics and relevance scores
- **🎨 Modern UI**: Beautiful Streamlit interface with responsive design

## 🏗️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB (local)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 2.5 Flash
- **Data Source**: [NU eLibrary](https://elibrary.nu.edu.om/)

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini 2.5 Flash)

## 🚀 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /Users/ekarshasumajk/Desktop/poc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   
   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your `.env` file

## 📊 Data Ingestion

Before running the app, you need to ingest the data into ChromaDB:

```bash
python data_ingestion.py
```

This will:
- Read all JSON and Markdown files from the data folders
- Extract faculty publication information
- Create embeddings using Sentence Transformers
- Store everything in a local ChromaDB database

Expected output:
```
🚀 Starting NU eLibrary Data Ingestion...
🗑️  Cleared existing collection
✅ Successfully loaded XXX documents into ChromaDB
📊 Collection Statistics:
   Total Documents: XXX
   Collection Name: nu_library_books
✨ Data ingestion complete!
```

## 🎯 Running the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 💡 Usage

1. **Enter a search query** in the search box
   - Example: "mechanical engineering research"
   - Example: "publications by Dr Ahmed"
   - Example: "faculty publications in pharmacy"

2. **Click Search** or press Enter

3. **View Results**:
   - AI-generated analysis of your query
   - Top 5 most relevant publications
   - AI-generated summaries for each publication
   - Relevance scores
   - Direct links to research profiles

## 📁 Project Structure

```
poc/
├── app.py                          # Main Streamlit application
├── data_ingestion.py               # Data loading and processing
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                           # Environment variables (create this)
├── chroma_db/                     # ChromaDB storage (auto-created)
├── f3061ffd-e619-46d4-8e9a-87385fa8e95f/     # JSON data folder
└── f3061ffd-e619-46d4-8e9a-87385fa8e95f 2/   # Markdown data folder
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

### Database Settings

- Database Path: `./chroma_db` (local directory)
- Collection Name: `nu_library_books`
- Embedding Model: `all-MiniLM-L6-v2` (runs locally, no API needed)

### Search Parameters

- Top K Results: 5 (configurable in `app.py`)
- Embedding Dimension: 384 (automatic)

## 🎨 Features in Detail

### Semantic Search
The system uses sentence transformers to create embeddings of both the query and documents, enabling semantic understanding beyond keyword matching.

### RAG Architecture
1. **Retrieval**: ChromaDB finds top 5 most relevant publications
2. **Augmentation**: Context is enriched with metadata and relevance scores
3. **Generation**: Gemini 2.5 Flash generates summaries and recommendations

### AI Summaries
For each publication, the system generates:
- Relevance explanation
- Author expertise highlights
- Publication context

## 📊 Data Source

All data is sourced from the official National University eLibrary:
- **URL**: https://elibrary.nu.edu.om/
- **Content**: Faculty publications from various colleges
- **Colleges**: COE, COMHS, COP, IMCO, and more

## 🔒 Privacy & Data

- All data processing happens locally
- No user data is stored
- Only embeddings are sent to Google for LLM processing
- Source data is from public university library

## 🐛 Troubleshooting

### "Database not found" error
Run `python data_ingestion.py` first to create the database.

### "GOOGLE_API_KEY not found" error
Make sure you've created a `.env` file with your API key.

### Slow search results
First search might be slow as models load. Subsequent searches will be faster.

### Import errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

## 🚀 Future Enhancements

- [ ] Add more filters (by college, author, year)
- [ ] Export search results to PDF
- [ ] Multi-language support
- [ ] Advanced query suggestions
- [ ] Bookmark favorite publications
- [ ] Citation export (BibTeX, APA, etc.)

## 📝 License

This project is created for educational purposes. All publication data belongs to National University, Oman.

## 🙏 Acknowledgments

- National University Libraries for providing the data
- Google for Gemini 2.5 Flash API
- ChromaDB team for the excellent vector database
- Streamlit for the amazing web framework

## 📧 Contact

For questions or issues, please refer to the [NU eLibrary](https://elibrary.nu.edu.om/) official website.

---

**Built with ❤️ using Gemini 2.5 Flash, ChromaDB, and Streamlit**
