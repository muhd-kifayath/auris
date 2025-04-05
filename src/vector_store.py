import chromadb
import numpy as np

class VectorStore:
    def __init__(self, collection_name="documents"):
        """Initialize ChromaDB and create a collection."""
        self.client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
        self.collection = self.client.get_or_create_collection(collection_name)

    def store_embeddings(self, embeddings, documents):
        """Store document embeddings in ChromaDB."""
        for i, (embedding, document) in enumerate(zip(embeddings, documents)):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedding.tolist()],
                metadatas=[{"text": document}],
            )

    def search(self, query_embedding, top_k=3):
        """Search for the closest documents."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        return [doc["text"] for doc in results["metadatas"][0]]

    def get_embedding_count(self):
        """Return the number of stored vectors."""
        return self.collection.count()
