"""
RAG Pipeline - Complete Retrieval-Augmented Generation System
Connects ChromaDB retrieval with Ollama LLM generation
"""

import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from typing import List, Dict, Optional
import time
from pathlib import Path


class UniversityRAG:
    def __init__(
        self,
        vectorstore_path="data/vectorstore",
        collection_name="university_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        top_k=5
    ):
        """
        Initialize the RAG system
        
        Args:
            vectorstore_path: Path to ChromaDB storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            llm_model: Ollama model for generation
            top_k: Number of chunks to retrieve
        """
        print("="*60)
        print("Initializing University RAG System")
        print("="*60)
        
        # Load embedding model
        print(f"\nLoading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Embedding model loaded")
        
        # Connect to ChromaDB
        print(f"\nConnecting to vector database...")
        self.client = chromadb.PersistentClient(path=vectorstore_path)
        self.collection = self.client.get_collection(collection_name)
        print(f"Connected to collection '{collection_name}'")
        print(f"  Total documents: {self.collection.count()}")
        
        # LLM settings
        self.llm_model = llm_model
        self.top_k = top_k
        
        # Verify Ollama model
        self._verify_llm()
        
        print("\n" + "="*60)
        print("RAG System Ready!")
        print("="*60 + "\n")
    
    def _verify_llm(self):
        """Verify Ollama model is available"""
        try:
            models = ollama.list()
            model_names = [model['model'] for model in models['models']]
            
            if not any(self.llm_model in name for name in model_names):
                print(f"\nWarning: Model '{self.llm_model}' not found!")
                print(f"Available models: {model_names}")
                print(f"Download with: ollama pull {self.llm_model}")
            else:
                print(f"\nLLM model '{self.llm_model}' is available")
        except Exception as e:
            print(f"\nCould not verify Ollama: {str(e)}")
    
    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant chunks from vector database
        
        Args:
            query: User's question
            n_results: Number of results to retrieve (defaults to self.top_k)
        
        Returns:
            List of retrieved chunks with metadata
        """
        if n_results is None:
            n_results = self.top_k
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        chunks = []
        for i in range(len(results['documents'][0])):
            chunks.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return chunks
    
    def generate(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks from vector database
        
        Returns:
            Dictionary with response and metadata
        """
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['metadata']['title']}]\n{chunk['document']}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are a helpful virtual assistant for Conestoga College.

Your role is to answer questions based ONLY on the provided context from the university's website.

Guidelines:
- Be friendly, professional, and helpful
- Answer only based on the context provided below
- If the context doesn't contain the answer, say: "I don't have that specific information in my knowledge base. I recommend contacting the admissions office or visiting the college website for more details."
- Keep answers clear and concise but complete
- When citing information, mention which source it came from (e.g., "According to the Admissions page...")
- Use bullet points for lists when appropriate
- Don't make up information not in the context"""
        
        full_prompt = f"""{system_prompt}

Context from Conestoga College website:

{context_text}

Student Question: {query}

Answer:"""
        
        # Generate response
        start_time = time.time()
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=full_prompt,
                options={
                    'temperature': 0.3,  # More focused answers
                    'num_predict': 500,  # Max tokens
                }
            )
            response_text = response['response']
        except Exception as e:
            response_text = f"I apologize, but I encountered an error: {str(e)}"
        
        elapsed = time.time() - start_time
        
        # Extract unique sources
        sources = []
        seen_urls = set()
        for chunk in context_chunks:
            url = chunk['metadata'].get('url', '')
            if url and url not in seen_urls:
                sources.append({
                    'title': chunk['metadata'].get('title', 'Unknown'),
                    'url': url,
                    'relevance': 1 - chunk['distance']  # Convert distance to similarity
                })
                seen_urls.add(url)
        
        return {
            'query': query,
            'answer': response_text,
            'sources': sources,
            'chunks_retrieved': len(context_chunks),
            'generation_time': elapsed
        }
    
    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            question: User's question
            verbose: If True, print detailed information
        
        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {question}")
            print(f"{'='*60}")
        
        # Step 1: Retrieve
        if verbose:
            print(f"\n[1/2] Retrieving relevant context...")
        
        retrieve_start = time.time()
        chunks = self.retrieve(question)
        retrieve_time = time.time() - retrieve_start
        
        if verbose:
            print(f"Retrieved {len(chunks)} chunks in {retrieve_time:.2f}s")
            print(f"\nTop sources:")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"  {i}. {chunk['metadata']['title']}")
                print(f"     Distance: {chunk['distance']:.4f}")
        
        # Step 2: Generate
        if verbose:
            print(f"\n[2/2] Generating answer...")
        
        result = self.generate(question, chunks)
        
        if verbose:
            print(f"Generated in {result['generation_time']:.2f}s")
            print(f"\n{'='*60}")
            print(f"Answer:")
            print(f"{'='*60}")
            print(result['answer'])
            print(f"\n{'='*60}")
            print(f"Sources:")
            print(f"{'='*60}")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']}")
                print(f"   {source['url']}")
                print(f"   Relevance: {source['relevance']:.2%}")
        
        # Add timing info
        result['retrieval_time'] = retrieve_time
        result['total_time'] = retrieve_time + result['generation_time']
        
        return result
    
    def chat(self, conversation_history: List[Dict], new_question: str) -> Dict:
        """
        Chat with conversation history
        
        Args:
            conversation_history: List of previous Q&A pairs
            new_question: New question from user
        
        Returns:
            Response dictionary
        """
        # For now, just use the latest question
        # In a more advanced version, you could use conversation history
        # to reformulate the query or maintain context
        
        return self.query(new_question)


# Test the RAG system
if __name__ == "__main__":
    # Initialize RAG system
    rag = UniversityRAG(
        vectorstore_path="data/vectorstore",
        collection_name="university_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        top_k=5
    )
    
    # Test queries
    test_questions = [
        "What are the admission requirements?",
        "What programs does the college offer?",
        "Tell me about student services",
        "How do I apply?",
        "What is the tuition cost?",
    ]
    
    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"Test {i}/{len(test_questions)}")
        print(f"{'#'*60}")
        
        result = rag.query(question, verbose=True)
        
        if i < len(test_questions):
            print("\n" + "-"*60)
            input("Press Enter to continue to next question...")
    
    print("\n\n" + "="*60)
    print("RAG System Testing Complete!")
    print("="*60)
    print("\nSystem Performance:")
    print(f"  Average retrieval time: ~{retrieve_time:.2f}s")
    print(f"  Average generation time: ~{result['generation_time']:.2f}s")
    print(f"  Average total time: ~{result['total_time']:.2f}s")
    print("="*60)