import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    def __init__(self, dim, index_path="vector_index.faiss", metadata_path = "metadata.pkl"):
        self.dim= dim
        self.index_path= Path(index_path)
        self.metadata_path= Path(metadata_path)

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def add(self, embedding, docs):
        embeddings = np.array(embedding).astype("float32")
        self.index.add(embeddings)
        # Ensure only simple dicts (avoid objects that cause recursion)
        cleaned_docs = []
        for d in docs:
            cleaned_docs.append({
                "source": d.get("source", ""),
                "page": d.get("page", None),
                "content": d.get("content", "")
            })
        self.metadata.extend(cleaned_docs)

    def save(self):
        faiss.write_index(self.index, str(self.metadata_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata,f)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "content":self.metadata[idx]["content"],
                "source":self.metadata[idx]["source"],
                "score":float(distances[0][i])
            })
        return results







