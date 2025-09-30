import requests
from bs4 import BeautifulSoup
import csv
import time
import urllib.parse
from urllib.parse import urljoin

keywords = [
"media studies", "communication studies", "journalism", "mass communication", "public relations",
"advertising", "social psychology", "developmental psychology", "cognitive psychology", "behavioral science",
"criminology", "criminal justice", "penology", "social research", "social theory",
"social change", "globalization", "development studies", "poverty alleviation", "inequality",
"social justice", "human development", "welfare economics", "behavioral economics", "institutional economics",
"political economy", "comparative politics", "governance", "democracy", "citizenship",
"civil society", "NGO management", "environmental science", "environmental management", "ecology",
"ecosystem", "biodiversity", "conservation", "wildlife management", "natural resources",
"resource management", "sustainability", "sustainable development", "climate change", "global warming",
"carbon footprint", "greenhouse gases", "environmental policy", "environmental economics", "green economy",
"circular economy", "waste management", "recycling", "pollution control", "air quality",
"water quality", "soil science", "land management", "forestry", "marine science",
"oceanography", "coastal management", "environmental assessment", "environmental impact", "environmental monitoring",
"renewable resources", "non-renewable resources", "energy conservation", "environmental education", "environmental awareness",
"ecological footprint", "environmental sustainability", "agriculture", "agronomy", "crop science",
"plant science", "horticulture", "agricultural engineering", "irrigation", "agricultural economics",
"farm management", "agricultural extension", "food science", "food technology", "food safety",
"food security", "nutrition science", "animal science", "livestock management", "poultry science",
"dairy science", "veterinary science", "agricultural research", "sustainable agriculture", "organic farming",
"precision agriculture", "agricultural biotechnology", "crop protection", "pest management", "integrated pest management",
"fertilizers", "agricultural marketing", "agribusiness", "food processing", "food preservation",
"food quality", "post harvest technology", "aquaculture", "fisheries", "agricultural policy",
"land use", "architecture", "architectural design", "urban design", "interior design",
"landscape architecture", "building design", "architectural theory", "architectural history", "sustainable architecture",
"green architecture", "architectural drawing", "architectural modeling", "building information modeling", "BIM",
"space planning", "design thinking", "creative design", "graphic design", "industrial design",
"product design", "design principles", "aesthetics", "form and function", "site analysis",
"building materials", "construction technology", "architectural engineering", "restoration", "heritage conservation",
"mathematics", "applied mathematics", "pure mathematics", "algebra", "calculus",
"differential equations", "linear algebra", "probability", "statistics", "mathematical statistics",
"biostatistics", "descriptive statistics", "inferential statistics", "hypothesis testing", "regression analysis",
"correlation", "ANOVA", "statistical modeling", "time series", "forecasting",
"sampling methods", "multivariate analysis", "factor analysis", "cluster analysis", "data analysis",
"statistical software", "SPSS", "R programming", "SAS", "mathematical modeling",
"optimization", "operations research", "numerical methods", "computational mathematics", "discrete mathematics",
"number theory", "geometry", "topology", "mathematical logic", "set theory",
"linguistics", "applied linguistics", "language acquisition", "second language learning", "phonetics",
"phonology", "morphology", "syntax", "semantics", "pragmatics",
"sociolinguistics", "psycholinguistics", "discourse analysis", "corpus linguistics", "translation studies",
"interpretation", "bilingualism", "multilingualism", "language teaching", "TESOL",
"English language teaching", "Arabic language", "language skills", "reading comprehension", "writing skills",
"listening skills", "speaking skills", "vocabulary", "grammar", "language assessment"
]

def scrape_library_books():
    """Scrape book data from elibrary.nu.edu.om for all keywords and save to CSV"""

    base_url = "https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    all_books = []

    for i, keyword in enumerate(keywords):
        print(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")

        # URL encode the keyword
        encoded_keyword = urllib.parse.quote(keyword)

        # Construct the search URL
        params = {
            'idx': '',
            'q': encoded_keyword,
            'limit': '',
            'weight_search': '1'
        }

        try:
            # Make the request
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.content, 'lxml')

            # Find the results container (table with id 'userresults' or similar)
            results_container = soup.find('div', {'id': 'userresults'})
            if not results_container:
                results_container = soup.find('table', {'id': 'userresults'})
            if not results_container:
                results_container = soup.find('table', class_=lambda x: x and 'table' in x.lower())

            if results_container:
                # Find book entries - they are table rows (tr) within the results container
                book_rows = results_container.find_all('tr')
                # Filter out header rows and rows without bibliocol
                book_rows = [row for row in book_rows if row.find('td', class_='bibliocol')]
            else:
                # Fallback: find all table rows that contain bibliocol
                book_rows = soup.find_all('tr', lambda tag: tag.find('td', class_='bibliocol'))

            print(f"Found {len(book_rows)} book entries for '{keyword}'")

            for row in book_rows:
                try:
                    # Extract book information
                    book_data = extract_book_info(row, base_url)
                    if book_data:
                        book_data['keyword'] = keyword
                        all_books.append(book_data)

                except Exception as e:
                    print(f"Error extracting book info: {e}")
                    continue

        except requests.RequestException as e:
            print(f"Error fetching data for keyword '{keyword}': {e}")
            continue

        # Rate limiting - be respectful to the server
        time.sleep(1.5)  # Slightly longer delay for full scrape

        # Save progress every 25 keywords for full scrape
        if (i + 1) % 25 == 0:
            save_to_csv(all_books, f'result/library_books_progress_{i+1}.csv')
            print(f"Progress saved after {i+1} keywords")

    # Save final results
    save_to_csv(all_books, 'result/library_books_final.csv')
    print(f"Scraping completed! Total books found: {len(all_books)}")

def extract_book_info(book_row, base_url):
    """Extract book information from a single book table row"""

    book_info = {
        'title': '',
        'link': '',
        'author': '',
        'description': '',
        'keyword': ''
    }

    try:
        # Find the bibliocol td which contains the book details
        biblio_col = book_row.find('td', class_='bibliocol')
        if not biblio_col:
            return None

        # Find title - in a.title link
        title_link = biblio_col.find('a', class_='title')
        if title_link:
            book_info['title'] = title_link.get_text(strip=True)
            # Make full URL
            book_info['link'] = urljoin(base_url, title_link['href'])

        # Find authors - in ul.author li a
        author_links = biblio_col.find_all('ul', class_='author')
        authors = []
        for author_ul in author_links:
            author_items = author_ul.find_all('li')
            for item in author_items:
                author_link = item.find('a')
                if author_link:
                    authors.append(author_link.get_text(strip=True))

        if authors:
            book_info['author'] = ', '.join(authors)

        # Find description - look for various description elements
        # First try to find any text that might be a description
        desc_parts = []

        # Look for publication details
        pub_div = biblio_col.find('div', class_='results_summary publisher')
        if pub_div:
            pub_text = pub_div.get_text(strip=True)
            # Remove the "Publication details:" label
            pub_text = pub_text.replace('Publication details:', '').strip()
            if pub_text:
                desc_parts.append(f"Publication: {pub_text}")

        # Look for edition info
        edition_div = biblio_col.find('div', class_='results_summary edition')
        if edition_div:
            edition_text = edition_div.get_text(strip=True)
            edition_text = edition_text.replace('Edition:', '').strip()
            if edition_text:
                desc_parts.append(f"Edition: {edition_text}")

        # Look for availability info (might give clues about the book)
        avail_div = biblio_col.find('div', class_='results_summary availability')
        if avail_div:
            avail_text = avail_div.get_text(strip=True)
            avail_text = avail_text.replace('Availability:', '').strip()
            if avail_text and len(avail_text) < 200:  # Only if not too long
                desc_parts.append(f"Availability: {avail_text}")

        # Combine description parts
        if desc_parts:
            book_info['description'] = ' | '.join(desc_parts)

        # If still no description, try to get other relevant text from the biblio col
        if not book_info['description']:
            all_text_elements = biblio_col.find_all(text=True)
            relevant_text = []
            for text in all_text_elements:
                text = text.strip()
                if len(text) > 10 and not text.startswith('by') and text != book_info['title']:
                    # Skip navigation text, labels, etc.
                    if not any(skip in text.lower() for skip in ['availability:', 'publication details:', 'edition:', 'items available']):
                        relevant_text.append(text)

            if relevant_text:
                book_info['description'] = ' '.join(relevant_text[:3])[:500]  # Take first 3 text elements, limit to 500 chars

    except Exception as e:
        print(f"Error in extract_book_info: {e}")

    # Only return if we have at least a title
    return book_info if book_info['title'] else None

def save_to_csv(books_data, filename):
    """Save book data to CSV file"""

    if not books_data:
        print("No data to save")
        return

    fieldnames = ['title', 'link', 'author', 'description', 'keyword']

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(books_data)

        print(f"Data saved to {filename} ({len(books_data)} books)")

    except Exception as e:
        print(f"Error saving to CSV: {e}")

def scrape_library_books_test(num_keywords=5):
    """Test version that only scrapes the first num_keywords"""

    base_url = "https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    all_books = []

    # Only use first num_keywords for testing
    test_keywords = keywords[:num_keywords]

    for i, keyword in enumerate(test_keywords):
        print(f"Processing keyword {i+1}/{len(test_keywords)}: {keyword}")

        # URL encode the keyword
        encoded_keyword = urllib.parse.quote(keyword)

        # Construct the search URL
        params = {
            'idx': '',
            'q': encoded_keyword,
            'limit': '',
            'weight_search': '1'
        }

        try:
            # Make the request
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.content, 'lxml')

            # Find the results container (table with id 'userresults' or similar)
            results_container = soup.find('div', {'id': 'userresults'})
            if not results_container:
                results_container = soup.find('table', {'id': 'userresults'})
            if not results_container:
                results_container = soup.find('table', class_=lambda x: x and 'table' in x.lower())

            if results_container:
                # Find book entries - they are table rows (tr) within the results container
                book_rows = results_container.find_all('tr')
                # Filter out header rows and rows without bibliocol
                book_rows = [row for row in book_rows if row.find('td', class_='bibliocol')]
            else:
                # Fallback: find all table rows that contain bibliocol
                book_rows = soup.find_all('tr', lambda tag: tag.find('td', class_='bibliocol'))

            print(f"Found {len(book_rows)} book entries for '{keyword}'")

            for row in book_rows:
                try:
                    # Extract book information
                    book_data = extract_book_info(row, base_url)
                    if book_data:
                        book_data['keyword'] = keyword
                        all_books.append(book_data)

                except Exception as e:
                    print(f"Error extracting book info: {e}")
                    continue

        except requests.RequestException as e:
            print(f"Error fetching data for keyword '{keyword}': {e}")
            continue

        # Rate limiting - be respectful to the server
        time.sleep(1)

        # Save progress every 2 keywords during testing
        if (i + 1) % 2 == 0:
            save_to_csv(all_books, f'result/library_books_test_progress_{i+1}.csv')
            print(f"Progress saved after {i+1} keywords")

    # Save final test results
    save_to_csv(all_books, 'result/library_books_test_final.csv')
    print(f"Test scraping completed! Total books found: {len(all_books)}")

if __name__ == "__main__":
    # Test with 10 keywords first
    scrape_library_books_test(210)
