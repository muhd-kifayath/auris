from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from sklearn.preprocessing import normalize


MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.documents = []
        self.dimension = 384  # MiniLM embedding size

        # Load index if it exists
        self._load_index()

    def vectorize(self, documents):
        """Creates or updates a FAISS index from given documents."""
        print(len(documents))
        new_embeddings = np.array([self.model.encode(doc) for doc in documents], dtype=np.float32).reshape(-1, self.dimension)
 
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # self.index.add(document_embeddings)
        
        self.index.add(new_embeddings)
        self.documents.extend(documents)

        # Save updated index
        self._save_index()

        return self.index, self.model

    def search(self, query_text, top_k=1):
        """Searches FAISS index for the most relevant document."""
        if self.index is None:
            raise ValueError("Index is empty. Please add documents first.")
        _, indices = self.index.search(query_embedding, top_k)
        query_embedding = np.array([self.model.encode(query_text)], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        
        return [self.documents[i] for i in indices[0]]

    def _save_index(self):
        """Saves FAISS index and metadata to disk."""
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.documents, f)

    def _load_index(self):
        """Loads FAISS index and metadata if available."""
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "rb") as f:
                self.documents = pickle.load(f)



# #📝 Add Documents
# from vectorizer import Vectorizer

# vec = Vectorizer()

# documents = [
#     "MRI is used in medical diagnostics.",
#     "Machine learning improves medical imaging.",
#     "Deep learning helps detect cancer."
# ]

# vec.vectorize(documents)
# print("Documents added successfully!")

# #🔍 Search for Similar Text
# query = "How does AI help in medical imaging?"
# result = vec.search(query, top_k=2)
# print("Best Matches:", result)

