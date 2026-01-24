"""
Quick Test for RAG Pipeline
Simple script to verify everything works
"""

from src.rag_pipeline import UniversityRAG

def main():
    print("\n" + "="*70)
    print(" "*20 + "RAG SYSTEM QUICK TEST")
    print("="*70)
    
    # Initialize the RAG system
    print("\n[Step 1] Initializing RAG system...")
    print("-"*70)
    
    rag = UniversityRAG(
        vectorstore_path="data/vectorstore",
        collection_name="university_knowledge",
        llm_model="llama3.2:3b",
        top_k=3  # Retrieve top 3 chunks
    )
    
    # Simple test question
    print("\n[Step 2] Testing with a sample question...")
    print("-"*70)
    
    question = "What programs are offered?"
    print(f"\nQuestion: {question}")
    print("\nProcessing...\n")
    
    # Get answer
    result = rag.query(question, verbose=False)
    
    # Display results
    print("="*70)
    print("ANSWER:")
    print("="*70)
    print(result['answer'])
    
    print("\n" + "="*70)
    print("SOURCES:")
    print("="*70)
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['title']}")
        print(f"   URL: {source['url']}")
        print(f"   Relevance: {source['relevance']:.1%}")
    
    print("\n" + "="*70)
    print("PERFORMANCE:")
    print("="*70)
    print(f"Retrieval time: {result['retrieval_time']:.2f}s")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Total time: {result['total_time']:.2f}s")
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE - Try your own questions!")
    print("="*70)
    print("(Type 'quit' to exit)\n")
    
    while True:
        try:
            user_question = input("\nYour question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_question:
                continue
            
            print("\nThinking...\n")
            result = rag.query(user_question, verbose=False)
            
            print("-"*70)
            print("Answer:")
            print("-"*70)
            print(result['answer'])
            
            print("\n" + "-"*70)
            print("Sources:")
            print("-"*70)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} - {source['url']}")
            
            print(f"\n(Answered in {result['total_time']:.2f}s)")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()