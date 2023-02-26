import faiss


class Faiss:
    """
    Vector index with Faiss 
    (See: https://www.pinecone.io/learn/faiss-tutorial/)
    """

    def __init__(self, d=768):
        self.index = faiss.IndexFlatIP(d)

    def add(self, documents_embeddings):
        self.index.add(documents_embeddings)

    def search(self, queries_embeddings, k):
        scores, indices = self.index.search(queries_embeddings, k)
        return scores, indices
