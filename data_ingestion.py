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
import csv


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
        file_metadata = self.extract_file_metadata(markdown_text, url)

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

                    # Store the raw content for embedding
                    book['raw_content'] = content

                    # Extract all structured data
                    self.extract_publication_details(book, content, file_metadata)

                    # Add book even with minimal info
                    if book:
                        books.append(book)

        return books

    def extract_file_metadata(self, markdown_text, url):
        """Extract metadata from the entire file"""
        metadata = {
            'source_url': url,
            'college': self.identify_college_from_url(url),
            'authors': [],
            'collections': [],
            'item_types': [],
            'search_results_count': 0
        }

        lines = markdown_text.split('\n')

        # Extract authors list
        in_authors = False
        for line in lines:
            if '### Authors' in line:
                in_authors = True
                continue
            elif in_authors and line.startswith('### '):
                break
            elif in_authors and line.strip().startswith('- ['):
                # Extract author name from markdown link
                author_match = re.search(r'\[([^\]]+)\]', line)
                if author_match:
                    metadata['authors'].append(author_match.group(1))

        # Extract collections
        in_collections = False
        for line in lines:
            if '### Collections' in line:
                in_collections = True
                continue
            elif in_collections and line.startswith('### '):
                break
            elif in_collections and line.strip().startswith('- ['):
                collection_match = re.search(r'\[([^\]]+)\]', line)
                if collection_match:
                    metadata['collections'].append(collection_match.group(1))

        # Extract item types
        in_item_types = False
        for line in lines:
            if '### Item types' in line:
                in_item_types = True
                continue
            elif in_item_types and line.startswith('### '):
                break
            elif in_item_types and line.strip().startswith('- ['):
                item_match = re.search(r'\[([^\]]+)\]', line)
                if item_match:
                    metadata['item_types'].append(item_match.group(1))

        # Extract search results count
        for line in lines:
            if 'Your search returned' in line:
                count_match = re.search(r'returned (\d+) results', line)
                if count_match:
                    metadata['search_results_count'] = int(count_match.group(1))
                break

        return metadata

    def extract_publication_details(self, book, content, file_metadata):
        """Extract detailed publication information"""
        # Extract author - multiple patterns for robustness
        author_patterns = [
            r'by<br>- \[([^\]]+)\]',  # Standard pattern
            r'\[([^]]+)\]\([^)]+\)',  # Fallback pattern
        ]

        for pattern in author_patterns:
            author_match = re.search(pattern, content)
            if author_match:
                author = author_match.group(1).replace('- ', '').strip()
                book['author'] = author
                break

        # Extract title - try to get more specific than "Faculty Publication"
        title_match = re.search(r'\[([^\]]+)\]\([^)]+\)', content.split('by<br>')[0] if 'by<br>' in content else content)
        if title_match and title_match.group(1) != 'Faculty Publication':
            book['title'] = title_match.group(1)
        else:
            book['title'] = 'Faculty Publication'

        # Extract detail URL
        detail_match = re.search(r'\[Faculty Publication\]\(([^)]+)\)', content)
        if detail_match:
            book['detail_url'] = detail_match.group(1)

        # Extract research links with better pattern matching
        research_links = {
            'google_scholar': r'\[Tab here for Google Scholar\]\(([^)]+)\)',
            'research_gate': r'\[Tab here for Research Gate\]\(([^)]+)\)',
            'scopus': r'\[Tab here for Scopus\]\(([^)]+)\)'
        }

        for link_type, pattern in research_links.items():
            match = re.search(pattern, content)
            if match:
                book[link_type] = match.group(1)

        # Extract college from multiple sources
        college = self.extract_college_info(content, file_metadata)
        if college:
            book['college'] = college

        # Extract availability information
        if 'Not for loan' in content:
            book['availability'] = 'Reference only - Not for loan'
        elif 'Available' in content:
            book['availability'] = 'Available for loan'

        # Extract item type
        if 'FP' in content or 'Faculty Publication' in content:
            book['item_type'] = 'Faculty Publication'
        elif 'LBK' in content or 'Library Book' in content:
            book['item_type'] = 'Library Book'

        # Add file-level metadata
        book['source_college'] = file_metadata.get('college')
        book['total_results_in_file'] = file_metadata.get('search_results_count')

    def identify_college_from_url(self, url):
        """Identify college from URL"""
        if 'COE' in url.upper():
            return 'College of Engineering'
        elif 'COMHS' in url.upper():
            return 'College of Medicine and Health Sciences'
        elif 'COP' in url.upper():
            return 'College of Pharmacy'
        elif 'IMCO' in url.upper():
            return 'International Maritime College Oman'
        return None

    def extract_college_info(self, content, file_metadata):
        """Extract college information from multiple sources"""
        # Check content for college mentions
        college_indicators = {
            'College of Engineering': ['College of Engineering', 'COE'],
            'College of Medicine and Health Sciences': ['College of Medicine', 'Medicine and Health Sciences', 'COMHS'],
            'College of Pharmacy': ['College of Pharmacy', 'COP'],
            'International Maritime College Oman': ['International Maritime', 'Maritime College', 'IMCO']
        }

        for college_name, indicators in college_indicators.items():
            for indicator in indicators:
                if indicator in content:
                    return college_name

        # Fallback to file metadata
        return file_metadata.get('college')
    
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
                if stripped.startswith('|') and 'Faculty Publication' in line and 'by<br>- [' in line:
                    # Split by pipe
                    columns = [col.strip() for col in stripped.split('|')]

                    if len(columns) > 3:
                        book = {}
                        content = columns[3]

                        # Extract author - pattern: by<br>- [Author Name](url)
                        author_match = re.search(r'by<br>- \[([^\]]+)\]', content)
                        if author_match:
                            book['author'] = author_match.group(1).strip()

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

                        # Extract college from URL or content
                        if 'COE' in url or 'College of Engineering' in content:
                            book['college'] = 'College of Engineering'
                        elif 'COMHS' in url or 'College of Medicine' in content:
                            book['college'] = 'College of Medicine and Health Sciences'
                        elif 'COP' in url or 'College of Pharmacy' in content:
                            book['college'] = 'College of Pharmacy'
                        elif 'IMCO' in url or 'International Maritime' in content:
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
    
    def load_data_from_folders(self, md_folder):
        """Load all data from Markdown folder with hierarchical chunking"""
        documents = []
        metadatas = []
        ids = []

        doc_id = 0

        # Process Markdown files only
        md_path = Path(md_folder)
        print(f"\n📁 Processing Markdown folder: {md_path.absolute()}")
        md_files = list(md_path.glob('*.md'))
        print(f"   Found {len(md_files)} Markdown files")

        # Group files by college for hierarchical chunking
        college_groups = {}

        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()

                file_metadata = self.extract_file_metadata(md_content, str(md_file))
                college = file_metadata.get('college', 'Unknown')

                if college not in college_groups:
                    college_groups[college] = []

                college_groups[college].append((md_file, md_content, file_metadata))

            except Exception as e:
                print(f"   ✗ Error processing {md_file.name}: {e}")

        # Process each college group hierarchically
        for college, file_data in college_groups.items():
            print(f"\n🏛️  Processing college: {college}")

            # Extract all publications for this college
            college_publications = []
            for md_file, md_content, file_metadata in file_data:
                books = self.extract_book_info_from_markdown(md_content, str(md_file))
                college_publications.extend(books)

                print(f"   ✓ {md_file.name}: extracted {len(books)} publications")

            # Create hierarchical chunks
            college_chunks = self.create_hierarchical_chunks(college, college_publications, file_data)

            # Add chunks to documents
            for chunk in college_chunks:
                documents.append(chunk['text'])

                # Only include primitive types in metadata (ChromaDB requirement)
                metadata = {
                    'source': 'markdown',
                    'college': college,
                    'chunk_type': chunk['type'],
                    'chunk_level': chunk['level'],
                    'url': 'https://elibrary.nu.edu.om/'
                }

                # Add primitive metadata fields
                chunk_meta = chunk.get('metadata', {})
                for key, value in chunk_meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, (list, dict)):
                        # Convert complex types to string representation
                        metadata[key] = str(value)

                metadatas.append(metadata)
                ids.append(f"doc_{doc_id}")
                doc_id += 1

        # Add all documents to ChromaDB
        if documents:
            print(f"\n💾 Adding {len(documents)} documents to ChromaDB...")

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
            print("   2. Do the folders contain .md files?")
            print("   3. Do the files contain valid publication data?")

        return len(documents)

    def load_data_from_csv(self, csv_file_path):
        """Load book data from CSV file and create hierarchical chunks"""
        documents = []
        metadatas = []
        ids = []

        doc_id = 0

        print(f"\n📁 Processing CSV file: {csv_file_path}")

        try:
            with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                books = list(reader)

            print(f"   Found {len(books)} book records")

            if not books:
                print("⚠️  No books found in CSV file")
                return 0

            # Group books by keyword for hierarchical chunking
            keyword_groups = {}
            for book in books:
                keyword = book.get('keyword', 'Unknown')
                if keyword not in keyword_groups:
                    keyword_groups[keyword] = []
                keyword_groups[keyword].append(book)

            # Process each keyword group hierarchically
            for keyword, keyword_books in keyword_groups.items():
                print(f"\n🏷️  Processing keyword: {keyword} ({len(keyword_books)} books)")

                # Create hierarchical chunks
                keyword_chunks = self.create_csv_hierarchical_chunks(keyword, keyword_books)

                # Add chunks to documents
                for chunk in keyword_chunks:
                    documents.append(chunk['text'])

                    # Only include primitive types in metadata (ChromaDB requirement)
                    metadata = {
                        'source': 'csv',
                        'keyword': keyword,
                        'chunk_type': chunk['type'],
                        'chunk_level': chunk['level'],
                        'url': 'https://elibrary.nu.edu.om/'
                    }

                    # Add primitive metadata fields
                    chunk_meta = chunk.get('metadata', {})
                    for key, value in chunk_meta.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        elif isinstance(value, (list, dict)):
                            # Convert complex types to string representation
                            metadata[key] = str(value)

                    metadatas.append(metadata)
                    ids.append(f"csv_doc_{doc_id}")
                    doc_id += 1

            # Add all documents to ChromaDB
            if documents:
                print(f"\n💾 Adding {len(documents)} documents to ChromaDB...")

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

            return len(documents)

        except Exception as e:
            print(f"   ✗ Error processing CSV file: {e}")
            return 0

    def create_csv_hierarchical_chunks(self, keyword, books):
        """Create hierarchical chunks for CSV data: keyword → author → individual books"""
        chunks = []

        if not books:
            return chunks

        # Level 1: Keyword-level summary chunk (biggest)
        keyword_summary = self.create_keyword_summary_chunk(keyword, books)
        chunks.append(keyword_summary)

        # Level 2: Author collection chunks (medium) - only if multiple books by same author
        author_groups = self.group_books_by_author(books)
        for author, author_books in author_groups.items():
            if len(author_books) > 1:  # Only create author chunks if they have multiple books
                author_chunk = self.create_author_summary_chunk_csv(author, author_books, keyword)
                chunks.append(author_chunk)

        # Level 3: Individual book chunks (smallest)
        for book in books:
            book_chunk = self.create_book_chunk(book, keyword)
            chunks.append(book_chunk)

        return chunks

    def create_keyword_summary_chunk(self, keyword, books):
        """Create a big chunk summarizing all books for a keyword"""
        total_books = len(books)
        authors = set(book.get('author', 'Unknown') for book in books if book.get('author'))

        # Aggregate statistics
        availability_stats = {}
        college_stats = {}

        for book in books:
            # Extract availability info
            description = book.get('description', '')
            if 'Not for loan' in description or 'Reference only' in description:
                availability = 'Reference only'
            elif 'Available for loan' in description:
                availability = 'Available for loan'
            else:
                availability = 'Unknown'
            availability_stats[availability] = availability_stats.get(availability, 0) + 1

            # Extract college info from description
            if 'College of Engineering' in description:
                college = 'College of Engineering'
            elif 'College of Medicine and Health Sciences' in description:
                college = 'College of Medicine and Health Sciences'
            elif 'College of Pharmacy' in description:
                college = 'College of Pharmacy'
            elif 'International Maritime College Oman' in description:
                college = 'International Maritime College Oman'
            else:
                college = 'Unknown'
            college_stats[college] = college_stats.get(college, 0) + 1

        # Create comprehensive summary text
        summary_parts = [
            f"KEYWORD OVERVIEW: {keyword}",
            f"Total Books: {total_books}",
            f"Total Authors: {len(authors)}",
            "",
            "COLLEGE DISTRIBUTION:"
        ]

        for college, count in college_stats.items():
            summary_parts.append(f"  - {college}: {count} books")

        summary_parts.extend(["", "AVAILABILITY:"])
        for availability, count in availability_stats.items():
            summary_parts.append(f"  - {availability}: {count} books")

        summary_parts.extend(["", "TOP AUTHORS:"])
        author_counts = {}
        for book in books:
            author = book.get('author', 'Unknown')
            author_counts[author] = author_counts.get(author, 0) + 1

        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for author, count in top_authors:
            summary_parts.append(f"  - {author}: {count} books")

        summary_text = "\n".join(summary_parts)

        return {
            'text': summary_text,
            'type': 'keyword_summary',
            'level': 1,
            'metadata': {
                'total_books': total_books,
                'total_authors': len(authors),
                'keyword': keyword,
                'college_stats': college_stats,
                'availability_stats': availability_stats
            }
        }

    def create_author_summary_chunk_csv(self, author, books, keyword):
        """Create a medium chunk summarizing an author's books for a keyword"""
        book_count = len(books)

        # Aggregate author statistics
        colleges = set()
        availability_types = set()

        for book in books:
            description = book.get('description', '')

            # Extract college
            if 'College of Engineering' in description:
                colleges.add('College of Engineering')
            elif 'College of Medicine and Health Sciences' in description:
                colleges.add('College of Medicine and Health Sciences')
            elif 'College of Pharmacy' in description:
                colleges.add('College of Pharmacy')
            elif 'International Maritime College Oman' in description:
                colleges.add('International Maritime College Oman')

            # Extract availability
            if 'Not for loan' in description or 'Reference only' in description:
                availability_types.add('Reference only')
            elif 'Available for loan' in description:
                availability_types.add('Available for loan')

        # Create author summary
        summary_parts = [
            f"AUTHOR PROFILE: {author}",
            f"Keyword: {keyword}",
            f"Total Books: {book_count}",
            f"Colleges: {', '.join(colleges) if colleges else 'Unknown'}",
            f"Availability Types: {', '.join(availability_types) if availability_types else 'Unknown'}",
            "",
            "BOOK TITLES:"
        ]

        # List book titles
        for i, book in enumerate(books[:10], 1):  # Limit to first 10 for chunk size
            title = book.get('title', 'Unknown Title')
            summary_parts.append(f"  {i}. {title}")

        if len(books) > 10:
            summary_parts.append(f"  ... and {len(books) - 10} more books")

        summary_text = "\n".join(summary_parts)

        return {
            'text': summary_text,
            'type': 'author_summary',
            'level': 2,
            'metadata': {
                'author': author,
                'keyword': keyword,
                'book_count': book_count,
                'colleges': list(colleges),
                'availability_types': list(availability_types)
            }
        }

    def create_book_chunk(self, book, keyword):
        """Create a small chunk for individual book details"""
        # Create document text
        doc_text = self.create_csv_document_text(book)

        # Add structured context
        context_parts = [
            f"BOOK DETAIL",
            f"Keyword: {keyword}",
            f"Author: {book.get('author', 'Unknown')}",
            f"Title: {book.get('title', 'Unknown Title')}",
        ]

        # Extract additional info from description
        description = book.get('description', '')
        if 'College of Engineering' in description:
            context_parts.append("College: College of Engineering")
        elif 'College of Medicine and Health Sciences' in description:
            context_parts.append("College: College of Medicine and Health Sciences")
        elif 'College of Pharmacy' in description:
            context_parts.append("College: College of Pharmacy")
        elif 'International Maritime College Oman' in description:
            context_parts.append("College: International Maritime College Oman")

        if 'Not for loan' in description or 'Reference only' in description:
            context_parts.append("Availability: Reference only")
        elif 'Available for loan' in description:
            context_parts.append("Availability: Available for loan")

        context_parts.extend(["", "CONTENT:"])
        context_text = "\n".join(context_parts)

        # Combine context with document text
        full_text = f"{context_text}\n\n{doc_text}"

        return {
            'text': full_text,
            'type': 'book_detail',
            'level': 3,
            'metadata': {
                'author': book.get('author'),
                'title': book.get('title'),
                'keyword': keyword,
                'link': book.get('link'),
                'description': book.get('description')
            }
        }

    def group_books_by_author(self, books):
        """Group books by author"""
        author_groups = {}
        for book in books:
            author = book.get('author', 'Unknown')
            if author not in author_groups:
                author_groups[author] = []
            author_groups[author].append(book)
        return author_groups

    def create_csv_document_text(self, book_info):
        """Create comprehensive document text for CSV book data"""
        text_parts = []

        # Start with core book information
        if 'author' in book_info and book_info['author']:
            author = book_info['author']
            text_parts.append(f"Author: {author}")
            # Add author variations for better searchability
            text_parts.append(f"Dr {author} Professor {author} faculty {author}")

        if 'title' in book_info and book_info['title']:
            text_parts.append(f"Title: {book_info['title']}")

        if 'keyword' in book_info and book_info['keyword']:
            text_parts.append(f"Keyword: {book_info['keyword']}")

        # Extract and add college information from description
        description = book_info.get('description', '')
        if 'College of Engineering' in description:
            text_parts.append("College: College of Engineering")
            text_parts.append("NU College of Engineering")
        elif 'College of Medicine and Health Sciences' in description:
            text_parts.append("College: College of Medicine and Health Sciences")
            text_parts.append("NU College of Medicine and Health Sciences")
        elif 'College of Pharmacy' in description:
            text_parts.append("College: College of Pharmacy")
            text_parts.append("NU College of Pharmacy")
        elif 'International Maritime College Oman' in description:
            text_parts.append("College: International Maritime College Oman")
            text_parts.append("NU International Maritime College Oman")

        # Extract availability information
        if 'Not for loan' in description or 'Reference only' in description:
            text_parts.append("Availability: Reference only - Not for loan")
        elif 'Available for loan' in description:
            text_parts.append("Availability: Available for loan")
        elif 'Checked out' in description:
            text_parts.append("Availability: Currently checked out")
        elif 'In transit' in description:
            text_parts.append("Availability: In transit")

        # Extract publication and edition info
        if 'Publication:' in description:
            pub_match = re.search(r'Publication:\s*([^|]+)', description)
            if pub_match:
                text_parts.append(f"Publication: {pub_match.group(1).strip()}")

        if 'Edition:' in description:
            edition_match = re.search(r'Edition:\s*([^|]+)', description)
            if edition_match:
                text_parts.append(f"Edition: {edition_match.group(1).strip()}")

        # Add link information
        if 'link' in book_info and book_info['link']:
            text_parts.append(f"Library Link: {book_info['link']}")

        # Add raw description for additional context
        if description:
            # Clean up description for better embedding
            clean_desc = description.replace('|', ' ').replace('  ', ' ')
            text_parts.append(f"Description: {clean_desc}")

        # Join all parts with newlines for better readability and embedding
        return "\n".join(text_parts)

    def create_hierarchical_chunks(self, college, publications, file_data):
        """Create hierarchical chunks: big chunks then split into smaller ones"""
        chunks = []

        if not publications:
            return chunks

        # Level 1: College-level summary chunk (biggest)
        college_summary = self.create_college_summary_chunk(college, publications, file_data)
        chunks.append(college_summary)

        # Level 2: Author collection chunks (medium)
        author_groups = self.group_publications_by_author(publications)
        for author, author_pubs in author_groups.items():
            if len(author_pubs) > 1:  # Only create author chunks if they have multiple publications
                author_chunk = self.create_author_summary_chunk(author, author_pubs, college)
                chunks.append(author_chunk)

        # Level 3: Individual publication chunks (smallest)
        for pub in publications:
            pub_chunk = self.create_publication_chunk(pub, college)
            chunks.append(pub_chunk)

        return chunks

    def create_college_summary_chunk(self, college, publications, file_data):
        """Create a big chunk summarizing the entire college's publications"""
        total_publications = len(publications)
        authors = set(pub.get('author', 'Unknown') for pub in publications if pub.get('author'))

        # Aggregate statistics
        item_types = {}
        availability_stats = {}
        research_links_count = 0

        for pub in publications:
            # Count item types
            item_type = pub.get('item_type', 'Unknown')
            item_types[item_type] = item_types.get(item_type, 0) + 1

            # Count availability
            availability = pub.get('availability', 'Unknown')
            availability_stats[availability] = availability_stats.get(availability, 0) + 1

            # Count research links
            if any(pub.get(link) for link in ['google_scholar', 'research_gate', 'scopus']):
                research_links_count += 1

        # Create comprehensive summary text
        summary_parts = [
            f"COLLEGE OVERVIEW: {college}",
            f"Total Publications: {total_publications}",
            f"Total Authors: {len(authors)}",
            f"Publications with Research Links: {research_links_count}",
            "",
            "ITEM TYPES:"
        ]

        for item_type, count in item_types.items():
            summary_parts.append(f"  - {item_type}: {count}")

        summary_parts.extend(["", "AVAILABILITY:"])
        for availability, count in availability_stats.items():
            summary_parts.append(f"  - {availability}: {count}")

        summary_parts.extend(["", "TOP AUTHORS:"])
        author_counts = {}
        for pub in publications:
            author = pub.get('author', 'Unknown')
            author_counts[author] = author_counts.get(author, 0) + 1

        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for author, count in top_authors:
            summary_parts.append(f"  - {author}: {count} publications")

        # Add file-level information
        if file_data:
            summary_parts.extend(["", "COLLECTIONS:"])
            all_collections = set()
            for _, _, file_metadata in file_data:
                all_collections.update(file_metadata.get('collections', []))
            for collection in sorted(all_collections):
                summary_parts.append(f"  - {collection}")

        summary_text = "\n".join(summary_parts)

        return {
            'text': summary_text,
            'type': 'college_summary',
            'level': 1,
            'metadata': {
                'total_publications': total_publications,
                'total_authors': len(authors),
                'college': college,
                'item_types': item_types,
                'availability_stats': availability_stats,
                'research_links_count': research_links_count
            }
        }

    def create_author_summary_chunk(self, author, publications, college):
        """Create a medium chunk summarizing an author's publications"""
        pub_count = len(publications)

        # Aggregate author statistics
        item_types = set(pub.get('item_type', 'Unknown') for pub in publications)
        has_research_links = any(
            any(pub.get(link) for link in ['google_scholar', 'research_gate', 'scopus'])
            for pub in publications
        )

        # Create author summary
        summary_parts = [
            f"AUTHOR PROFILE: {author}",
            f"College: {college}",
            f"Total Publications: {pub_count}",
            f"Item Types: {', '.join(item_types)}",
            f"Has Research Links: {'Yes' if has_research_links else 'No'}",
            "",
            "PUBLICATIONS:"
        ]

        # List publication titles/IDs
        for i, pub in enumerate(publications[:10], 1):  # Limit to first 10 for chunk size
            title = pub.get('title', 'Faculty Publication')
            availability = pub.get('availability', 'Unknown')
            summary_parts.append(f"  {i}. {title} ({availability})")

        if len(publications) > 10:
            summary_parts.append(f"  ... and {len(publications) - 10} more publications")

        summary_text = "\n".join(summary_parts)

        return {
            'text': summary_text,
            'type': 'author_summary',
            'level': 2,
            'metadata': {
                'author': author,
                'college': college,
                'publication_count': pub_count,
                'item_types': list(item_types),
                'has_research_links': has_research_links
            }
        }

    def create_publication_chunk(self, publication, college):
        """Create a small chunk for individual publication details"""
        # Use enhanced document text creation
        doc_text = self.create_document_text(publication)

        # Add structured context
        context_parts = [
            f"PUBLICATION DETAIL",
            f"College: {college}",
            f"Author: {publication.get('author', 'Unknown')}",
            f"Title: {publication.get('title', 'Faculty Publication')}",
            f"Item Type: {publication.get('item_type', 'Unknown')}",
            f"Availability: {publication.get('availability', 'Unknown')}",
        ]

        # Add research links if available
        research_links = []
        if publication.get('google_scholar'):
            research_links.append(f"Google Scholar: {publication['google_scholar']}")
        if publication.get('research_gate'):
            research_links.append(f"Research Gate: {publication['research_gate']}")
        if publication.get('scopus'):
            research_links.append(f"Scopus: {publication['scopus']}")

        if research_links:
            context_parts.append("Research Links:")
            context_parts.extend(f"  - {link}" for link in research_links)

        context_parts.extend(["", "CONTENT:"])
        context_text = "\n".join(context_parts)

        # Combine context with document text
        full_text = f"{context_text}\n\n{doc_text}"

        return {
            'text': full_text,
            'type': 'publication_detail',
            'level': 3,
            'metadata': {
                'author': publication.get('author'),
                'title': publication.get('title'),
                'college': college,
                'item_type': publication.get('item_type'),
                'availability': publication.get('availability'),
                'detail_url': publication.get('detail_url'),
                'google_scholar': publication.get('google_scholar'),
                'research_gate': publication.get('research_gate'),
                'scopus': publication.get('scopus')
            }
        }

    def group_publications_by_author(self, publications):
        """Group publications by author"""
        author_groups = {}
        for pub in publications:
            author = pub.get('author', 'Unknown')
            if author not in author_groups:
                author_groups[author] = []
            author_groups[author].append(pub)
        return author_groups
    
    def create_document_text(self, book_info):
        """Create comprehensive document text for embedding with all structured data"""
        text_parts = []

        # Start with core publication information
        if 'author' in book_info and book_info['author']:
            author = book_info['author']
            text_parts.append(f"Author: {author}")

            # Add author variations for better searchability
            text_parts.append(f"Dr {author} Professor {author} faculty {author}")

        if 'title' in book_info and book_info['title']:
            text_parts.append(f"Title: {book_info['title']}")

        if 'college' in book_info and book_info['college']:
            college = book_info['college']
            text_parts.append(f"College: {college}")
            text_parts.append(f"NU {college}")  # Add variations for search

        if 'item_type' in book_info and book_info['item_type']:
            text_parts.append(f"Type: {book_info['item_type']}")

        if 'availability' in book_info and book_info['availability']:
            text_parts.append(f"Availability: {book_info['availability']}")

        # Add research links information
        research_links = []
        if book_info.get('google_scholar'):
            research_links.append("Google Scholar profile available")
        if book_info.get('research_gate'):
            research_links.append("Research Gate profile available")
        if book_info.get('scopus'):
            research_links.append("Scopus profile available")

        if research_links:
            text_parts.append(f"Research Links: {', '.join(research_links)}")

        # Add raw content if available (cleaned up)
        if 'raw_content' in book_info:
            raw_text = book_info['raw_content']

            # Remove markdown links but keep the text content
            raw_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', raw_text)

            # Clean up HTML breaks and extra whitespace
            raw_text = raw_text.replace('<br>', ' ')
            raw_text = re.sub(r'\s+', ' ', raw_text).strip()

            text_parts.append(f"Content: {raw_text}")

        # Add any additional metadata
        if 'source_college' in book_info and book_info['source_college']:
            text_parts.append(f"Source College: {book_info['source_college']}")

        if 'total_results_in_file' in book_info:
            text_parts.append(f"Total publications in search: {book_info['total_results_in_file']}")

        # Join all parts with newlines for better readability and embedding
        return "\n".join(text_parts)
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }


def main():
    """Main function to run data ingestion from CSV file with hierarchical chunking"""
    print("🚀 Starting NU eLibrary Data Ingestion...")
    print("   (Hierarchical chunking: Keyword → Author → Book levels)")

    # Path to CSV data file (relative to current directory)
    csv_file = "./library_books_test_progress_408.csv"

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

    # Load data from CSV file only
    doc_count = ingestion.load_data_from_csv(csv_file)

    # Print stats
    stats = ingestion.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Collection Name: {stats['collection_name']}")
    print("\n✨ Data ingestion complete!")


if __name__ == "__main__":
    main()
