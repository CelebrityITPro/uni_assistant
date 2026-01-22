"""
Text Processing & Chunking
Prepares scraped data for embedding and vector storage
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize text processor
        
        Args:
            chunk_size: Target character size for each chunk
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.processing_time = None
    
    def load_scraped_data(self, filepath="data/raw/scraped_data.json"):
        """Load scraped web pages"""
        print(f"Loading scraped data from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} pages")
        return data
    
    def load_pdf_data(self, filepath="data/raw/documents/processed_documents.json"):
        """Load processed PDF documents (if available)"""
        pdf_path = Path(filepath)
        
        if not pdf_path.exists():
            print(f"No PDF data found at {filepath} (skipping)")
            return []
        
        print(f"Loading PDF data from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} PDF documents")
        return data
    
    def chunk_text(self, text: str, metadata: dict) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk (url, title, etc.)
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        text_length = len(text)
        
        # If text is smaller than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            return [{
                'text': text,
                'metadata': metadata,
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Get chunk
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary (period followed by space)
            if end < text_length:
                # Look for last period in chunk
                last_period = chunk_text.rfind('. ')
                if last_period > self.chunk_size * 0.5:  # Only if it's not too early
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': metadata.copy(),
                'chunk_index': chunk_index,
                'char_start': start,
                'char_end': end
            })
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        # Add total chunks to metadata
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    def process_all_data(self, web_data, pdf_data=None):
        """
        Process all data (web pages and PDFs) into chunks
        
        Args:
            web_data: List of scraped web pages
            pdf_data: List of processed PDFs (optional)
        
        Returns:
            List of all chunks with metadata
        """
        self.start_time = datetime.now()
        all_chunks = []
        
        print(f"\n{'='*60}")
        print("Processing and Chunking Data")
        print(f"{'='*60}\n")
        
        # Process web pages
        print(f"Processing {len(web_data)} web pages...")
        
        for page in web_data:
            metadata = {
                'source_type': 'webpage',
                'url': page['url'],
                'title': page['title'],
                'domain': page.get('domain', ''),
                'scraped_at': page.get('scraped_at', '')
            }
            
            # Chunk the page content
            page_chunks = self.chunk_text(page['content'], metadata)
            all_chunks.extend(page_chunks)
        
        print(f"Created {len(all_chunks)} chunks from web pages")
        
        # Process PDFs (if available)
        if pdf_data:
            print(f"\nProcessing {len(pdf_data)} PDF documents...")
            pdf_start = len(all_chunks)
            
            for doc in pdf_data:
                metadata = {
                    'source_type': 'pdf',
                    'url': doc['url'],
                    'title': doc['title'],
                    'num_pages': doc.get('num_pages', 0),
                    'processed_at': doc.get('processed_at', '')
                }
                
                # Chunk the PDF content
                doc_chunks = self.chunk_text(doc['content'], metadata)
                all_chunks.extend(doc_chunks)
            
            pdf_chunks = len(all_chunks) - pdf_start
            print(f"Created {pdf_chunks} chunks from PDFs")
        
        print(f"\n{'='*60}")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"{'='*60}\n")
        
        # Calculate statistics
        chunk_lengths = [len(chunk['text']) for chunk in all_chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        print(f"Statistics:")
        print(f"  Average chunk size: {avg_length:.0f} characters")
        print(f"  Smallest chunk: {min(chunk_lengths)} characters")
        print(f"  Largest chunk: {max(chunk_lengths)} characters")
        
        self.chunks = all_chunks
        self.processing_time = datetime.now() - self.start_time
        return all_chunks
    
    def save_chunks(self, output_path="data/processed/chunks.json"):
        """Save processed chunks to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving chunks to {output_path}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.chunks)} chunks")
        
        if self.processing_time:
            print(f"Processing time: {self.processing_time}")
        
        # Save summary
        summary = {
            'total_chunks': len(self.chunks),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'processing_time_seconds': self.processing_time.total_seconds() if self.processing_time else 0,
            'processed_at': datetime.now().isoformat(),
            'source_types': {},
            'domains': {}
        }
        
        # Count by source type and domain
        for chunk in self.chunks:
            source_type = chunk['metadata'].get('source_type', 'unknown')
            summary['source_types'][source_type] = summary['source_types'].get(source_type, 0) + 1
            
            domain = chunk['metadata'].get('domain', 'unknown')
            if domain:
                summary['domains'][domain] = summary['domains'].get(domain, 0) + 1
        
        summary_path = output_file.parent / "chunks_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {summary_path}")


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = TextProcessor(
        chunk_size=1000,      # 1000 characters per chunk
        chunk_overlap=200     # 200 character overlap
    )
    
    # Load data
    web_data = processor.load_scraped_data("data/raw/scraped_data.json")
    pdf_data = processor.load_pdf_data("data/raw/documents/processed_documents.json")
    
    # Process and chunk everything
    chunks = processor.process_all_data(web_data, pdf_data)
    
    # Save processed chunks
    processor.save_chunks("data/processed/chunks.json")
    
    print("\n" + "="*60)
    print("Text Processing Complete!")
    print("="*60)
    print("Next Steps:")
    print("1. Review data/processed/chunks.json")
    print("2. Check data/processed/chunks_summary.json")
    print("3. Move to embedding generation (next script)")
    print("="*60)