from vectorizer import Vectorizer
from vector_store import VectorStore
from gpthelper import GPTHelper
from sklearn.preprocessing import normalize
import numpy as np
import os

# Load OpenAI API Key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store your API key in an environment variable

# Step 1: Initialize Vectorizer, VectorStore, and GPT
vectorizer = Vectorizer()
store = VectorStore()
# gpt = GPTHelper(api_key=OPENAI_API_KEY)

print(f"Number of vectors in database: {store.get_embedding_count()}")


# Step 2: Query Text
query_text = "What are the technical enhancements in baking machine?"
query_embedding = np.array([vectorizer.model.encode(query_text)], dtype=np.float32)
query_embedding = normalize(query_embedding, axis=1)

# Step 3: Search in Vector Database
results = store.search(query_embedding, top_k=1)

# Step 4: Generate GPT Response
if results:
    print("🔍 Retrieved Documents:")
    print("Length of results:", len(results))
    for res in results:
        print(res)
    
    # Ask GPT based on retrieved documents
    # answer = gpt.ask_gpt(query_text, results)
    # print("\n💡 GPT Response:")
    # print(answer)
else:
    print("⚠️ No relevant results found.")
