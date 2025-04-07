import os
import numpy as np
from sklearn.preprocessing import normalize
from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from vector_store import VectorStore

class FolderProcessor:
    def __init__(self, processor, vectorizer, store):
        self.processor = processor
        self.vectorizer = vectorizer
        self.store = store

    def read_all_files(self, folder_path):
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if filename.lower().endswith(('.pdf', '.docx', '.jpg', '.png', '.jpeg')):
                    files.append(file_path)
        return files

    def process_and_store(self, root_folder):
        file_paths = self.read_all_files(root_folder)
        all_documents = []
        for file_path in file_paths:
            # try:
            text = self.processor.process_document(file_path)
            if isinstance(text, list):
                all_documents.extend(text)
            else:
                all_documents.append(text)
            # except Exception as e:
            #     print(f"Failed to process {file_path}: {e}")

        print(f"Processed {len(all_documents)} documents. Creating embeddings...")

        # Vectorization
        index, model = self.vectorizer.vectorize(all_documents)
        embeddings = np.array([model.encode(doc) for doc in all_documents], dtype=np.float32)
        embeddings = normalize(embeddings, axis=1)

        # Store in vector DB
        self.store.store_embeddings(embeddings, all_documents)
        print("All embeddings stored in vector DB.")

processor = DocumentProcessor()
vectorizer = Vectorizer()
store = VectorStore()

folder_path = "documents"  # Specify your folder path here

runner = FolderProcessor(processor, vectorizer, store)
runner.process_and_store(folder_path)
