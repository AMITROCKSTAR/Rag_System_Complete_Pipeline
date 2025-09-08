from sentence_transformers import SentenceTransformer
class Embedder:
    def __init__(self, provider="sbert", model_name="all-MiniLM-L6-v2"):
        self.provider = provider
        if provider == "sbert":
            self.model = SentenceTransformer(model_name)

        else:
            raise ValueError("Sbert only supported here")


    def emb(self, texts: list):
        """
        Input: list of strings
        Output: list of embeddings
        """

        if self.provider == "sbert":
            #
           return self.model.encode(texts, convert_to_numpy=True).tolist()


