"""
University Website Scraper - Multi-Domain Version
Scrapes multiple university-related websites and saves content for RAG indexing
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import os
from datetime import datetime
from pathlib import Path


class MultiDomainUniversityScraper:
    def __init__(self, allowed_domains, output_dir="data/raw"):
        """
        Initialize the multi-domain scraper
        
        Args:
            allowed_domains: List of domains to scrape 
            output_dir: Where to save scraped data
        """
        self.allowed_domains = [domain.lower() for domain in allowed_domains]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track visited URLs to avoid duplicates
        self.visited_urls = set()
        self.scraped_data = []
        
        # Log file for skipped pages
        self.skipped_log = self.output_dir / "skipped_pages.log"
        self.max_pages = None
        self.start_time = None
        self.attempted = 0
        
        # Scraping delay settings
        self.delay = 2  # seconds between requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot)'
        }
        
        print(f"Initialized scraper for domains: {', '.join(self.allowed_domains)}")
    
    def is_valid_url(self, url):
        """Check if URL should be scraped"""
        parsed = urlparse(url)
        
        # Check if domain is in our allowed list
        domain = parsed.netloc.lower()
        if not any(allowed in domain for allowed in self.allowed_domains):
            return False
        
        # Skip certain file types
        skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.docx', 
                          '.xlsx', '.ppt', '.mp4', '.mp3', '.avi']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip login/admin pages
        skip_paths = ['/login', '/admin', '/user/', '/cart', '/wp-admin', 
                     '/signin', '/signup', '/register', '/checkout']
        if any(path in url.lower() for path in skip_paths):
            return False
        
        # Skip external links and social media
        skip_domains = ['facebook.com', 'twitter.com', 'instagram.com', 
                       'linkedin.com', 'youtube.com', 'mailto:']
        if any(skip in url.lower() for skip in skip_domains):
            return False
        
        return True
    
    def extract_content(self, soup):
        """Extract meaningful content from page"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=['content', 'main-content', 'page-content']) or
            soup.find('div', id=['content', 'main-content', 'page-content'])
        )
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def scrape_page(self, url):
        """Scrape a single page"""
        try:
            self.attempted += 1
            if self.attempted % 10 == 0:
                print(f"(scraping {self.attempted}/{self.max_pages})")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            text = self.extract_content(soup)
            
            # Only save if we got meaningful content
            if len(text) < 30:
                with open(self.skipped_log, 'a', encoding='utf-8') as f:
                    f.write(f"{url} - content too short ({len(text)} chars)\n")
                return []
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc['content'] if meta_desc and meta_desc.get('content') else ""
            
            # Extract links for further crawling
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                # Remove fragments and query params for deduplication
                clean_url = absolute_url.split('#')[0].split('?')[0]
                if self.is_valid_url(clean_url):
                    links.append(clean_url)
            
            # Remove duplicate links
            links = list(set(links))
            
            # Save page data
            page_data = {
                'url': url,
                'domain': urlparse(url).netloc,
                'title': title,
                'description': description,
                'content': text,
                'content_length': len(text),
                'scraped_at': datetime.now().isoformat(),
                'outgoing_links_count': len(links)
            }
            
            self.scraped_data.append(page_data)
            
            return links
            
        except requests.exceptions.Timeout:
            print(f"Timeout - skipping")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return []
    
    def crawl(self, start_urls, max_pages=50):
        """
        Crawl multiple websites starting from given URLs
        
        Args:
            start_urls: List of URLs to start crawling from
            max_pages: Maximum number of pages to scrape
        """
        to_visit = list(start_urls)
        pages_scraped = 0
        pages_failed = 0
        
        self.max_pages = max_pages
        self.start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"Starting multi-domain crawl")
        print(f"Allowed domains: {', '.join(self.allowed_domains)}")
        print(f"Seed URLs: {len(start_urls)}")
        print(f"Max pages: {max_pages}")
        print(f"{'='*60}\n")
        
        while to_visit and pages_scraped < max_pages:
            url = to_visit.pop(0)
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            
            # Scrape the page
            new_links = self.scrape_page(url)
            
            if new_links is not None and len(self.scraped_data) > pages_scraped:
                pages_scraped += 1
                
                # Add new links to visit queue
                for link in new_links:
                    if link not in self.visited_urls and link not in to_visit:
                        to_visit.append(link)
            else:
                pages_failed += 1
            
            # Be polite - wait between requests
            time.sleep(self.delay)
        
        print(f"\n{'='*60}")
        print(f"Crawl Complete!")
        print(f"{'='*60}")
        print(f"Total pages scraped: {pages_scraped}")
        print(f"Total pages failed: {pages_failed}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        
        # Summary by domain
        print(f"\nPages by domain:")
        domain_counts = {}
        for page in self.scraped_data:
            domain = page['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        for domain, count in sorted(domain_counts.items()):
            print(f"  - {domain}: {count} pages")
        
        self.remaining_in_queue = len(to_visit)
        return self.scraped_data
    
    def save_data(self, filename="scraped_data.json"):
        """Save scraped data to JSON file"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Data saved to: {filepath}")
        print(f"Total pages: {len(self.scraped_data)}")
        
        # Calculate statistics
        total_chars = sum(page['content_length'] for page in self.scraped_data)
        avg_chars = total_chars / len(self.scraped_data) if self.scraped_data else 0
        
        print(f"Total content: {total_chars:,} characters")
        print(f"Average per page: {avg_chars:.0f} characters")
        
        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        print(f"Total time elapsed: {elapsed}")
        print(f"Remaining pages in queue: {self.remaining_in_queue}")
        
        # Save a detailed summary
        summary = {
            'total_pages': len(self.scraped_data),
            'total_characters': total_chars,
            'average_characters': avg_chars,
            'total_time_elapsed_seconds': elapsed.total_seconds(),
            'remaining_pages_in_queue': self.remaining_in_queue,
            'scraped_at': datetime.now().isoformat(),
            'allowed_domains': self.allowed_domains,
            'pages_by_domain': {},
            'urls': [page['url'] for page in self.scraped_data]
        }
        
        # Count by domain
        for page in self.scraped_data:
            domain = page['domain']
            if domain not in summary['pages_by_domain']:
                summary['pages_by_domain'][domain] = 0
            summary['pages_by_domain'][domain] += 1
        
        summary_path = self.output_dir / "scrape_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}\n")



if __name__ == "__main__":
    # University domains to scrape
    ALLOWED_DOMAINS = [
        "conestogac.on.ca",           # Main site
        "conestogastudents.com",       # Student body site
        # Add more domains as needed
        # "library.conestogac.on.ca",
        # "careers.conestogac.on.ca",
    ]
    
    # Define important starting pages across all domains
    start_urls = [
        # Main site
        "https://www.conestogac.on.ca/",
        "https://www.conestogac.on.ca/future-students/",
        "https://www.conestogac.on.ca/international/",
        "https://www.conestogac.on.ca/admissions/",
        "https://www.conestogac.on.ca/campus-life-and-services/",
        "https://www.conestogac.on.ca/about/",
        "https://www.conestogac.on.ca/programs-and-courses/",
        
        # Student portal
        "https://www.conestogastudents.com/",
        "https://www.conestogastudents.com/about-us",
        "https://www.conestogastudents.com/getinvolved",
        "https://www.conestogastudents.com/studentlife",
        "https://www.conestogastudents.com/wellness",
        "https://www.conestogastudents.com/representation",


        
        # Add more starting points
    ]
    
    # Create scraper
    scraper = MultiDomainUniversityScraper(
        allowed_domains=ALLOWED_DOMAINS,
        output_dir="data/raw"
    )
    
    # Start crawling
    # Increase max_pages to get more content
    data = scraper.crawl(start_urls, max_pages=5000)
    
    # Save the data
    scraper.save_data()
    
    print("\n" + "="*60)