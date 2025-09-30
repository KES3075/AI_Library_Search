"""
Data ingestion script for NU eLibrary
Loads JSON and Markdown data into ChromaDB for RAG
"""
import json
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import re


class DataIngestion:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB client with local persistence"""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB with sentence transformer embeddings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use sentence transformers for embeddings (local, no API needed)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="nu_library_books",
            embedding_function=self.embedding_function,
            metadata={"description": "NU eLibrary faculty publications and books"}
        )
    
    def extract_book_info_from_markdown(self, markdown_text, url):
        """Extract book/publication information from markdown content"""
        books = []
        
        # The data is in markdown table format
        # Each table row contains publication info in the 4th column
        lines = markdown_text.split('\n')
        
        for line in lines:
            # Look for table rows with Faculty Publication
            stripped = line.strip()
            if stripped.startswith('|') and 'Faculty Publication' in line and 'by<br>' in line:
                # Split by pipe
                columns = [col.strip() for col in stripped.split('|')]
                
                if len(columns) > 3:
                    book = {}
                    content = columns[3]
                    
                    # Extract author - simpler pattern
                    try:
                        author_match = re.search(r'\[([^]]+)\]\([^)]+\)', content.split('by<br>')[1] if 'by<br>' in content else '')
                        if author_match:
                            book['author'] = author_match.group(1).replace('- ', '').strip()
                    except:
                        pass
                    
                    # Extract title
                    book['title'] = 'Faculty Publication'
                    
                    # Extract detail URL
                    detail_match = re.search(r'\[Faculty Publication\]\(([^)]+)\)', content)
                    if detail_match:
                        book['detail_url'] = detail_match.group(1)
                    
                    # Extract research links
                    if 'Google Scholar' in content:
                        scholar_match = re.search(r'\[Tab here for Google Scholar\]\(([^)]+)\)', content)
                        if scholar_match:
                            book['google_scholar'] = scholar_match.group(1)
                    
                    if 'Research Gate' in content:
                        rg_match = re.search(r'\[Tab here for Research Gate\]\(([^)]+)\)', content)
                        if rg_match:
                            book['research_gate'] = rg_match.group(1)
                    
                    if 'Scopus' in content:
                        scopus_match = re.search(r'\[Tab here for Scopus\]\(([^)]+)\)', content)
                        if scopus_match:
                            book['scopus'] = scopus_match.group(1)
                    
                    # Extract college
                    if 'College of Engineering' in content:
                        book['college'] = 'College of Engineering'
                    elif 'College of Medicine' in content:
                        book['college'] = 'College of Medicine and Health Sciences'
                    elif 'College of Pharmacy' in content:
                        book['college'] = 'College of Pharmacy'
                    elif 'International Maritime' in content:
                        book['college'] = 'International Maritime College Oman'
                    
                    # Availability
                    if 'Not for loan' in content:
                        book['availability'] = 'Reference only - Not for loan'
                    
                    # Add book even with minimal info
                    if book:
                        books.append(book)
        
        return books
    
    def extract_book_info_from_json(self, json_data, url):
        """Extract book/publication information from JSON content"""
        books = []
        
        # Extract from markdown field in JSON
        if 'markdown' in json_data:
            markdown_text = json_data['markdown']
            lines = markdown_text.split('\n')
            
            for line in lines:
                # Look for table rows with Faculty Publication
                stripped = line.strip()
                if stripped.startswith('|') and 'Faculty Publication' in line and 'by<br>' in line:
                    # Split by pipe
                    columns = [col.strip() for col in stripped.split('|')]
                    
                    if len(columns) > 3:
                        book = {}
                        content = columns[3]
                        
                        # Extract author - simpler pattern
                        author_match = re.search(r'\[([^]]+)\]\([^)]+\)', content.split('by<br>')[1] if 'by<br>' in content else '')
                        if author_match:
                            book['author'] = author_match.group(1).replace('- ', '').strip()
                        
                        # Extract title
                        book['title'] = 'Faculty Publication'
                        
                        # Extract detail URL
                        detail_match = re.search(r'\[Faculty Publication\]\(([^)]+)\)', content)
                        if detail_match:
                            book['detail_url'] = detail_match.group(1)
                        
                        # Extract research links
                        if 'Google Scholar' in content:
                            scholar_match = re.search(r'\[Tab here for Google Scholar\]\(([^)]+)\)', content)
                            if scholar_match:
                                book['google_scholar'] = scholar_match.group(1)
                        
                        if 'Research Gate' in content:
                            rg_match = re.search(r'\[Tab here for Research Gate\]\(([^)]+)\)', content)
                            if rg_match:
                                book['research_gate'] = rg_match.group(1)
                        
                        if 'Scopus' in content:
                            scopus_match = re.search(r'\[Tab here for Scopus\]\(([^)]+)\)', content)
                            if scopus_match:
                                book['scopus'] = scopus_match.group(1)
                        
                        # Extract college
                        if 'College of Engineering' in content:
                            book['college'] = 'College of Engineering'
                        elif 'College of Medicine' in content:
                            book['college'] = 'College of Medicine and Health Sciences'
                        elif 'College of Pharmacy' in content:
                            book['college'] = 'College of Pharmacy'
                        elif 'International Maritime' in content:
                            book['college'] = 'International Maritime College Oman'
                        
                        # Availability
                        if 'Not for loan' in content:
                            book['availability'] = 'Reference only - Not for loan'
                        
                        # Add book even with minimal info
                        if book:
                            books.append(book)
        
        # Add summary from JSON if available
        if 'summary' in json_data and books:
            for book in books:
                if 'summary' not in book:
                    book['summary'] = json_data['summary'][:500]
        
        return books
    
    def load_data_from_folders(self, json_folder, md_folder):
        """Load all data from JSON and Markdown folders"""
        documents = []
        metadatas = []
        ids = []
        
        doc_id = 0
        
        # Process JSON files
        json_path = Path(json_folder)
        print(f"\n📁 Processing JSON folder: {json_path.absolute()}")
        json_files = list(json_path.glob('*.json'))
        print(f"   Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                url = data.get('metadata', {}).get('url', str(json_file))
                books = self.extract_book_info_from_json(data, url)
                
                for book in books:
                    # Create a rich text representation for embedding
                    doc_text = self.create_document_text(book)
                    
                    # Add all documents without strict validation
                    documents.append(doc_text)
                    
                    # Add metadata
                    metadata = {
                        'source': 'json',
                        'file': json_file.name,
                        'url': url,
                        **book
                    }
                    metadatas.append(metadata)
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1
                
                if books:
                    print(f"   ✓ {json_file.name}: extracted {len(books)} publications")
                
            except Exception as e:
                print(f"   ✗ Error processing {json_file.name}: {e}")
        
        # Process Markdown files
        md_path = Path(md_folder)
        print(f"\n📁 Processing Markdown folder: {md_path.absolute()}")
        md_files = list(md_path.glob('*.md'))
        print(f"   Found {len(md_files)} Markdown files")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                books = self.extract_book_info_from_markdown(md_content, str(md_file))
                
                for book in books:
                    doc_text = self.create_document_text(book)
                    
                    # Add all documents
                    documents.append(doc_text)
                    
                    metadata = {
                        'source': 'markdown',
                        'file': md_file.name,
                        'url': 'https://elibrary.nu.edu.om/',
                        **book
                    }
                    metadatas.append(metadata)
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1
                
                if books:
                    print(f"   ✓ {md_file.name}: extracted {len(books)} publications")
                
            except Exception as e:
                print(f"   ✗ Error processing {md_file.name}: {e}")
        
        # Add all documents to ChromaDB
        if documents:
            # Count by source
            json_count = sum(1 for m in metadatas if m.get('source') == 'json')
            md_count = sum(1 for m in metadatas if m.get('source') == 'markdown')
            
            print(f"\n💾 Adding {len(documents)} documents to ChromaDB...")
            print(f"   - From JSON files: {json_count}")
            print(f"   - From Markdown files: {md_count}")
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            print(f"✅ Successfully loaded {len(documents)} documents into ChromaDB")
        else:
            print("\n⚠️  No documents found to load")
            print("   Please check:")
            print("   1. Are the folder paths correct?")
            print("   2. Do the folders contain .json and .md files?")
            print("   3. Do the files contain valid publication data?")
        
        return len(documents)
    
    def create_document_text(self, book_info):
        """Create a rich text representation of book for embedding"""
        parts = []
        
        if 'title' in book_info:
            parts.append(f"Title: {book_info['title']}")
        
        if 'author' in book_info:
            parts.append(f"Author: {book_info['author']}")
        
        if 'college' in book_info:
            parts.append(f"College: {book_info['college']}")
        
        if 'item_type' in book_info:
            parts.append(f"Type: {book_info['item_type']}")
        
        if 'summary' in book_info:
            parts.append(f"Summary: {book_info['summary']}")
        
        if 'availability' in book_info:
            parts.append(f"Availability: {book_info['availability']}")
        
        # Add research profile info
        profiles = []
        if 'google_scholar' in book_info:
            profiles.append("Google Scholar profile available")
        if 'research_gate' in book_info:
            profiles.append("ResearchGate profile available")
        if 'scopus' in book_info:
            profiles.append("Scopus indexed")
        
        if profiles:
            parts.append("Research Profiles: " + ", ".join(profiles))
        
        # Add a searchable description
        desc_parts = []
        if 'author' in book_info:
            desc_parts.append(f"Faculty publication by {book_info['author']}")
        if 'college' in book_info:
            desc_parts.append(f"from {book_info['college']}")
        
        if desc_parts:
            parts.append("Description: " + " ".join(desc_parts))
        
        return "\n".join(parts)
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }


def main():
    """Main function to run data ingestion"""
    print("🚀 Starting NU eLibrary Data Ingestion...")
    print("   (Processing both JSON and Markdown files)")
    
    # Paths to data folders (relative to current directory)
    json_folder = "./f3061ffd-e619-46d4-8e9a-87385fa8e95f"
    md_folder = "./f3061ffd-e619-46d4-8e9a-87385fa8e95f 2"
    
    # Initialize and load data
    ingestion = DataIngestion()
    
    # Clear existing collection (optional - comment out to keep existing data)
    try:
        ingestion.client.delete_collection("nu_library_books")
        ingestion.collection = ingestion.client.create_collection(
            name="nu_library_books",
            embedding_function=ingestion.embedding_function
        )
        print("🗑️  Cleared existing collection")
    except:
        pass
    
    # Load data from both folders
    doc_count = ingestion.load_data_from_folders(json_folder, md_folder)
    
    # Print stats
    stats = ingestion.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Collection Name: {stats['collection_name']}")
    print("\n✨ Data ingestion complete!")


if __name__ == "__main__":
    main()
