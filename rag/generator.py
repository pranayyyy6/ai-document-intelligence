 
# rag/generator.py
# PURPOSE: Generate answers using LLM + retrieved context
# WHY: The "G" in RAG — Generation using grounded context
# INTERVIEW: "How do you prevent hallucinations in your system?"

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq client once
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question: str, context: str) -> dict:
    """
    Generate answer using LLM with retrieved context.
    
    KEY DESIGN DECISIONS:
    
    1. System prompt tells LLM to ONLY use provided context
       WHY: Prevents hallucination — LLM can't make things up
    
    2. "If not in context, say I don't know"
       WHY: Honest system > confident wrong system
       Production RAG systems MUST have this
    
    3. Ask for page citation in answer
       WHY: Users trust answers more with sources
       INTERVIEW: "How do you handle explainability?"
    
    4. temperature=0.1
       WHY: Low temperature = focused, factual answers
       High temperature = creative, potentially wrong answers
       For Q&A systems always use low temperature
    """
    
    system_prompt = f"""You are a precise document assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
Always mention which page/source you found the answer in.
Be concise and accurate.

Context:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,     # Low = factual
        max_tokens=500        # Limit response length
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "model_used": "llama-3.1-8b-instant",
        "tokens_used": response.usage.total_tokens
    }