from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from vector_store import VectorStore
from sklearn.preprocessing import normalize
import numpy as np

# Step 1: Process the document
file_path = "documents/file.docx"
processor = DocumentProcessor()
processed_text = processor.process_document(file_path)

# Step 2: Convert text to embeddings
documents = processed_text
vectorizer = Vectorizer()
index, model = vectorizer.vectorize(documents)

# Step 3: Store in vector database
store = VectorStore()
document_embeddings = np.array([model.encode(doc) for doc in documents], dtype=np.float32)
document_embeddings = normalize(document_embeddings, axis=1)
store.store_embeddings(document_embeddings, documents)

print("✅ Document processed and stored successfully!")

# # Step 4: Query example
# query_text = "What are the technical enhancements in baking machine?"
# query_embedding = np.array([model.encode(query_text)], dtype=np.float32)
# query_embedding = normalize(query_embedding, axis=1)
# results = store.search(query_embedding, top_k=3)
# print(f"Query embedding shape: {query_embedding.shape}")
# print(f"Stored embedding shape: {document_embeddings.shape}")

# if results:
#     print("🔍 Query Results:")
#     for res in results:
#         print(res)
# else:
#     print("⚠️ No relevant results found.")