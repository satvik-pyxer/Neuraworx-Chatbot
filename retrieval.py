import os
from typing import List, Dict, Any, Optional
import numpy as np

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ingestion import COLLECTION_NAME, EMBEDDING_MODEL, get_qdrant_client, get_embedder
from model import get_llm

# --- Constants ---
TOP_K = 10  # Number of initial chunks to retrieve
FINAL_K = 3  # Number of chunks after re-ranking

# --- Query Processing ---
def optimize_query(query: str, chat_history: List[Dict[str, str]] = None) -> str:
    """
    Optimize the user query by considering the chat history.
    Uses an LLM to rewrite the query for better retrieval.
    """
    if not chat_history or len(chat_history) == 0:
        return query
    
    # Get the LLM
    llm = get_llm()
    
    # Prepare the prompt - updated format for Llama 3.1
    prompt = f"""
You are an AI assistant optimizing search queries for a RAG system.
Given the following chat history and the most recent user query, rewrite the query to be more effective for vector retrieval.
The optimized query should be standalone and capture the full context of what the user is asking about based on the conversation history.

CHAT HISTORY:
{format_chat_history(chat_history)}

MOST RECENT QUERY: {query}

OPTIMIZED QUERY (respond with the optimized query only, no additional text):

"""
    
    # Get the optimized query from the LLM
    optimized_query = llm.generate(prompt, max_tokens=200).strip()
    
    # If the LLM returned more than the query, try to extract just the query
    if len(optimized_query.split('\n')) > 1 or len(optimized_query) > len(query) * 3:
        # If it's too verbose, fall back to the original query
        return query
    
    return optimized_query

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history into a readable string for the LLM."""
    formatted = []
    for i, msg in enumerate(chat_history):
        if msg.get('role') and msg.get('content'):
            formatted.append(f"{msg['role'].upper()}: {msg['content']}")
    
    return "\n".join(formatted)

# --- Retrieval ---
def retrieve_chunks(query: str, chat_history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a query.
    
    1. Optimize the query based on chat history
    2. Retrieve initial candidates
    3. Re-rank the candidates
    4. Return the top results
    """
    # Optimize query
    optimized_query = optimize_query(query, chat_history)
    print(f"Optimized query: {optimized_query}")
    
    # Get embedding for query
    embedder = get_embedder()
    query_embedding = embedder.encode(optimized_query).tolist()
    
    # Retrieve initial candidates from Qdrant
    qdrant = get_qdrant_client()
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=TOP_K
    )
    
    # Extract chunks with their scores
    candidates = [
        {
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": hit.score
        }
        for hit in search_result
    ]
    
    # Re-rank
    reranked_results = rerank_chunks(optimized_query, candidates)
    
    # Return the top results
    return reranked_results[:FINAL_K]

def rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank chunks based on the query using a more sophisticated approach.
    """
    # If we have few or no chunks, return as is
    if len(chunks) <= 1:
        return chunks
    
    llm = get_llm()
    
    # Prepare scoring prompt - updated format for Llama 3.1
    base_prompt = f"""
You are an AI assistant evaluating the relevance of text chunks for a query.

QUERY: {query}

Rate each chunk on a scale of 1-100 based on how relevant and useful it is for answering the query.
Important factors:
- Exact answer presence (0-40 points)
- Supporting details (0-30 points) 
- Information quality (0-20 points)
- Content freshness (0-10 points)

Your rating should strictly be a single number (1-100) with no explanation.

YOUR RATING:
"""
    
    # Score each chunk
    for chunk in chunks:
        prompt = f"{base_prompt}\n\nTEXT CHUNK:\n{chunk['text']}\n\nRATING: "
        try:
            rating_str = llm.generate(prompt, max_tokens=10).strip()
            # Extract just the numeric rating
            numeric_chars = ''.join(c for c in rating_str if c.isdigit())
            if numeric_chars:
                rating = int(numeric_chars[:3])
                rating = min(100, max(1, rating))  # Ensure rating is between 1-100
                chunk["rerank_score"] = rating
            else:
                # If no numeric rating was found, use the original score
                chunk["rerank_score"] = chunk["score"] * 100
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fall back to the original score if reranking fails
            chunk["rerank_score"] = chunk["score"] * 100
    
    # Sort by rerank score
    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked

def generate_answer(query: str, chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
    """
    Generate an answer based on the retrieved chunks and query.
    """
    llm = get_llm()
    
    # Prepare context from chunks
    chunk_context = ""
    for i, chunk in enumerate(chunks):
        chunk_context += f"\n--- Chunk {i+1} [Source: {chunk['metadata'].get('filename', 'Unknown')}] ---\n"
        chunk_context += chunk["text"]
    
    # Prepare chat history context
    history_context = ""
    if chat_history and len(chat_history) > 0:
        history_context = "Previous conversation:\n" + format_chat_history(chat_history)
    
    # Prepare the prompt - updated format for Llama 3.1
    prompt = f"""
You are a helpful AI assistant answering questions based on retrieved information.
Guidelines:

1. Use only the information provided in the retrieved text to answer questions
2. Present your response in a natural, conversational manner
3. Do not mention or refer to the chunks or sources in your response
4. If the retrieved information doesn't contain the answer, clearly state that you don't have enough information to answer the question
5. Never fabricate information or make assumptions beyond what's explicitly stated in the retrieved text
6. Focus solely on answering what was asked without unnecessary explanations about your sourcing

HISTORY:
{history_context}

RETRIEVED INFORMATION:
{chunk_context}

USER QUERY: {query}

YOUR ANSWER:
"""
    
    # Generate the answer
    answer = llm.generate(prompt, max_tokens=1000)
    return answer