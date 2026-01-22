"""
Embedding Generation & Vector Store
Converts text chunks to embeddings and stores in ChromaDB
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime


class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2", vectorstore_path="data/vectorstore"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model
            vectorstore_path: Path to store ChromaDB
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded (dimension: {self.model.get_sentence_embedding_dimension()})")
        
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        print(f"\nInitializing ChromaDB at {vectorstore_path}...")
        self.client = chromadb.PersistentClient(path=str(self.vectorstore_path))
        print("ChromaDB initialized")
        
        self.collection = None
        self.embedding_time = None
    
    def create_collection(self, collection_name="university_knowledge", reset=False):
        """
        Create or get ChromaDB collection
        
        Args:
            collection_name: Name of the collection
            reset: If True, delete existing collection and create new one
        """
        if reset:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted existing collection: {collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "University knowledge base for RAG"}
        )
        
        existing_count = self.collection.count()
        print(f"Collection '{collection_name}' ready (existing documents: {existing_count})")
        
        return self.collection
    
    def load_chunks(self, chunks_path="data/processed/chunks.json"):
        """Load processed chunks"""
        print(f"\nLoading chunks from {chunks_path}...")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, chunks, batch_size=32):
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of text chunks with metadata
            batch_size: Number of chunks to process at once
        
        Returns:
            List of embeddings
        """
        print(f"\n{'='*60}")
        print(f"Generating Embeddings")
        print(f"{'='*60}\n")
        print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        self.embedding_start = datetime.now()
        
        # Generate embeddings with progress bar
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"\nGenerated {len(embeddings)} embeddings")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        
        self.embedding_time = datetime.now() - self.embedding_start
        
        return embeddings
    
    def store_in_vectordb(self, chunks, embeddings):
        """
        Store chunks and embeddings in ChromaDB
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors
        """
        print(f"\n{'='*60}")
        print(f"Storing in Vector Database")
        print(f"{'='*60}\n")
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = []
        
        # Prepare metadata (ChromaDB requires flat dictionaries)
        for i, chunk in enumerate(chunks):
            metadata = {
                'source_type': chunk['metadata'].get('source_type', ''),
                'url': chunk['metadata'].get('url', ''),
                'title': chunk['metadata'].get('title', ''),
                'domain': chunk['metadata'].get('domain', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
            }
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        print(f"Adding documents in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing in ChromaDB"):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"\nStored {len(chunks)} chunks in vector database")
        print(f"  Collection size: {self.collection.count()}")
    
    def test_retrieval(self, query="What are the admission requirements?", n_results=3):
        """
        Test the vector database with a sample query
        
        Args:
            query: Test query
            n_results: Number of results to retrieve
        """
        print(f"\n{'='*60}")
        print(f"Testing Retrieval")
        print(f"{'='*60}\n")
        print(f"Query: '{query}'")
        print(f"Retrieving top {n_results} results...\n")
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Display results
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"Result {i}:")
            print(f"  Source: {metadata.get('title', 'Unknown')}")
            print(f"  URL: {metadata.get('url', 'Unknown')}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Text preview: {doc[:200]}...")
            print()
        
        print("Retrieval test complete!")
    
    def save_stats(self):
        """Save statistics about the vector store"""
        if self.embedding_time:
            print(f"Embedding time: {self.embedding_time}")
        
        stats = {
            'collection_name': self.collection.name,
            'total_documents': self.collection.count(),
            'embedding_model': self.model.get_sentence_embedding_dimension(),
            'embedding_time_seconds': self.embedding_time.total_seconds() if self.embedding_time else 0,
            'created_at': datetime.now().isoformat(),
            'vectorstore_path': str(self.vectorstore_path)
        }
        
        stats_path = self.vectorstore_path / "vectorstore_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nStatistics saved to {stats_path}")


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",  # Lightweight model
        vectorstore_path="data/vectorstore"
    )
    
    # Create/reset collection
    embedder.create_collection(
        collection_name="university_knowledge",
        reset=True  # Set to False if you want to keep existing data
    )
    
    # Load processed chunks
    chunks = embedder.load_chunks("data/processed/chunks.json")
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(chunks, batch_size=32)
    
    # Store in ChromaDB
    embedder.store_in_vectordb(chunks, embeddings)
    
    # Save statistics
    embedder.save_stats()
    
    # Test retrieval
    print("\n" + "="*60)
    print("Testing the Vector Store")
    print("="*60)
    
    test_queries = [
        "What are the admission requirements?",
        "Tell me about student services",
        "How do I apply?",
    ]
    
    for query in test_queries:
        embedder.test_retrieval(query, n_results=2)
        print("-" * 60)
    
    print("\n" + "="*60)
    print("Embedding & Vector Store Setup Complete!")
    print("="*60)