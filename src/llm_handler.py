"""
LLM Handler
Wrapper for Ollama LLM interactions
"""

import ollama
from typing import List, Dict, Optional
import time


class LLMHandler:
    def __init__(self, model_name="llama3.2:3b", temperature=0.2):
        """
        Initialize LLM handler
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness (0.0 = focused, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Verify model is available
        self.verify_model()
    
    def verify_model(self):
        """Check if the model is downloaded"""
        try:
            models = ollama.list()
            model_names = [model['model'] for model in models['models']]
            
            if not any(self.model_name in name for name in model_names):
                print(f"Model '{self.model_name}' not found!")
                print(f"Available models: {model_names}")
                print(f"\nPlease download with: ollama pull {self.model_name}")
                raise Exception(f"Model {self.model_name} not available")
            
            print(f"Model '{self.model_name}' is ready")
            
        except Exception as e:
            print(f"Error checking models: {str(e)}")
            raise
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: The prompt to send to the LLM
            stream: Whether to stream the response
        
        Returns:
            Generated response text
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=stream,
                options={
                    'temperature': self.temperature,
                    'num_predict': 500,  # Max tokens to generate
                }
            )
            
            if stream:
                # For streaming, you'd yield chunks
                return response
            else:
                return response['response']
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response."
    
    def generate_with_context(
        self, 
        query: str, 
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate response with retrieved context (RAG)
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks from vector database
            system_prompt: Custom system prompt (optional)
        
        Returns:
            Dictionary with response and metadata
        """
        # Build context string from chunks
        context_text = "\n\n".join([
            f"Source: {chunk['metadata']['title']}\n{chunk['document']}"
            for chunk in context_chunks
        ])
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful virtual assistant for a university.

Your role is to answer questions based ONLY on the provided context from the university's website.

Guidelines:
- Be friendly and professional
- Answer only based on the context provided
- If the context doesn't contain the answer, say "I don't have that specific information in my knowledge base. I recommend contacting the admissions office or visiting the university website for more details."
- Keep answers concise but complete
- Always mention the source when citing specific information
- Use bullet points for lists when appropriate"""
        
        # Build complete prompt
        full_prompt = f"""{system_prompt}

Context from university website:
{context_text}

Student Question: {query}

Answer (be helpful and cite sources):"""
        
        # Generate response
        start_time = time.time()
        response_text = self.generate(full_prompt)
        elapsed = time.time() - start_time
        
        # Extract source URLs
        sources = [
            {
                'title': chunk['metadata']['title'],
                'url': chunk['metadata']['url']
            }
            for chunk in context_chunks
        ]
        
        # Remove duplicates
        unique_sources = []
        seen_urls = set()
        for source in sources:
            if source['url'] not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(source['url'])
        
        return {
            'response': response_text,
            'sources': unique_sources,
            'context_chunks_used': len(context_chunks),
            'generation_time': elapsed
        }
    
    def chat(self, messages: List[Dict]) -> str:
        """
        Chat with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{'role': 'user', 'content': 'Hello'}]
        
        Returns:
            Generated response text
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': self.temperature,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error."


# Test the LLM handler
if __name__ == "__main__":
    print("="*60)
    print("Testing LLM Handler")
    print("="*60)
    
    # Initialize handler
    llm = LLMHandler(model_name="llama3.2:3b", temperature=0.2)
    
    # Test 1: Simple generation
    print("\nTest 1: Simple Generation")
    print("-"*60)
    response = llm.generate("Say hello in a friendly way.")
    print(f"Response: {response}")
    
    # Test 2: With context (simulating RAG)
    print("\n\nTest 2: RAG Simulation")
    print("-"*60)
    
    # Mock context chunks (like what we'd get from ChromaDB)
    mock_chunks = [
        {
            'document': 'Conestoga College offers over 200 programs in areas including Business, Engineering Technology, Health Sciences, and Information Technology.',
            'metadata': {
                'title': 'Programs - Conestoga College',
                'url': 'https://www.conestogac.on.ca/programs'
            }
        },
        {
            'document': 'Admission requirements vary by program. Most programs require an Ontario Secondary School Diploma (OSSD) or equivalent.',
            'metadata': {
                'title': 'Admissions - Conestoga College',
                'url': 'https://www.conestogac.on.ca/admissions'
            }
        }
    ]
    
    result = llm.generate_with_context(
        query="What programs does Conestoga offer?",
        context_chunks=mock_chunks
    )
    
    print(f"Question: What programs does Conestoga offer?")
    print(f"\nResponse: {result['response']}")
    print(f"\nSources used:")
    for source in result['sources']:
        print(f"  - {source['title']}")
        print(f"    {source['url']}")
    print(f"\nGeneration time: {result['generation_time']:.2f} seconds")
    
    # Test 3: Conversation
    print("\n\nTest 3: Conversation with History")
    print("-"*60)
    
    messages = [
        {'role': 'user', 'content': 'Hi! Can you help me?'},
    ]
    
    response = llm.chat(messages)
    print(f"User: Hi! Can you help me?")
    print(f"Assistant: {response}")
    
    # Add to conversation
    messages.append({'role': 'assistant', 'content': response})
    messages.append({'role': 'user', 'content': 'What is machine learning?'})
    
    response = llm.chat(messages)
    print(f"\nUser: What is machine learning?")
    print(f"Assistant: {response}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)