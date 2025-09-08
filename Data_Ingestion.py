import os
from pathlib import Path
import docx2txt
import fitz
from bs4 import BeautifulSoup
from streamlit.string_util import clean_text
import requests
import trafilatura
import re
import tiktoken
from Vector_db_Retrieval import VectorStore
from Embedding import Embedder

class UnifiedIngestion:
    def __init__(self,input_dir: str=None,urls:list=None):
        self.input_dir = Path(input_dir) if input_dir else None
        self.urls=urls if urls else None
        print("Path",input_dir)

    #### Main Data Loader
    def load(self):
        all_docs=[]

        # Load Local files
        if self.input_dir:
            for file in self.input_dir.glob("*"):
                text = self.load_documents(file)
                if text:
                    all_docs.extend(text)


        for url in self.urls:
            text = self.load_web(url)
            if text:
                all_docs.append(text)

        return all_docs


    def load_documents(self,file_path):
        """Load all files from directory"""
        try:

            if file_path.suffix.lower() == ".pdf":
                return self._load_pdf(file_path)
            elif file_path.suffix.lower() == ".docx":
                return self.clean_text(docx2txt.process(file_path))
            elif file_path.suffix.lower() in [".html",".htm"]:
                return self.clean_text(self._load_html(file_path))

            else:
                print(f"Skipping unsupported file : {file_path}")
                return None
        except Exception as e:
            print(f"Failed to load {file_path}:{e}")
            return None

    def _load_pdf(self,file_path):
        """Extract text from PDF using PyMuPdf"""
        docs =[]
        with fitz.open(file_path) as doc:
            for i,page in enumerate(doc,start=1):
                text += page.get_text("text")
                if text.strip():
                    docs.append({
                        "source":str(file_path),
                        "page":i,
                        "content":self.clean_text(text)
                    })

        return docs

    def _load_docx(self,file_path):
        """Extract text from docx"""
        text = docx2txt.process(file_path)
        return self._clean_text(text)

    def _load_html(self,file_path):
        """Extract text from html"""
        with open(file_path,"r",encoding = "utf-8") as f:
            soup = BeautifulSoup(f,"html.parser")
            text = soup.get_text(seperator=" ")
        return text



    ## Web data loader

    def load_web(self,url):
        try:

            downloaded = trafilatura.fetch_url(url)
            extracted = trafilatura.extract(downloaded, include_links= True, include_formatting= True)
            if extracted:
                return {
                    "source":url,
                    "title": self._get_page_title(url),
                    "content": self.clean_text(extracted)
                }
            # Fallback: BS4
            response = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            soup = Beautifulsoup(response.text,"html.parser")
            text = soup.get_text(seperator=" ")
            return {"source":url, "title":soup.title.string if soup.title else None,"content":self.clean_text(text)}

        except Exception as e:
            print(f"Faailed to scrap {url}:{e}")

    def _get_page_title(self,url):
        try:
            response = requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
            soup = BeautifulSoup(response.text,"html.parser")
            return soup.title.string if soup.title else None

        except:
            return None



    def clean_text(self,text:str)-> str:
        """Basic text cleaning"""
        text = text.replace("\n", " ").replace("\t"," ")
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = " ".join(text.split()) # remove extra spaces
        return text.strip()

    def chunk_text(self, text, chunk_size=500, overlap=0.2, model_name="gpt-4o"):
        """
        Split text into chunks of ~chunk_size tokens with overlap.
        Returns list of chunks (text strings).
        """
        # Initialize tokenizer for the model
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)

        step = int(chunk_size * (1 - overlap))
        chunks = []

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            if i + chunk_size >= len(tokens):
                break  # last chunk
        return chunks
    # chunk all loaded documents while preserving metadata:
    def chunk_documents(self,all_documents, chunk_size=500, overlap=0.2, model_name="gpt-4o" ):
        """
        Input: list of documents from load()
        Output: list of chunks with metadata:
        {"source": ..., "page": ..., "title": ..., "content": chunk_text}
        """

        chunked_docs=[]

        for doc in all_documents:
            content = doc['content']
            chunks = self.chunk_text(content, chunk_size, overlap, model_name)
            for chunk in chunks:
                chunked_docs.append(
                    {
                        "source":doc.get("source"),
                        "page":doc.get("page"),
                        "title":doc.get("title"),
                        "content":chunk

                    }
                )
        return chunked_docs

if __name__== "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://www.anthropic.com/claude"
    ]
    ingestion = UnifiedIngestion(input_dir=r"C:\Users\amitr\Desktop\Rag_System_Complete_Pipeline",urls=urls)
    All_documents = ingestion.load()


    # for i,doc in enumerate(All_documents[:3],1):
    #     print(f"\n-------Document {i}-------\n")
    #     print(doc['content'][:500],"...\n")
    #     print(f"Source: {doc.get('source')}")
    #     if doc.get("page"):
    #         print(f"Document page: {doc['page']}")
    #     if doc.get('title'):
    #         print(f"Document title: {doc['title']}")
    #
    #     print(f"Content preview:", doc['content'][:300],"....\n")


    ## Chunk all docs
    chunked_docs= ingestion.chunk_documents(All_documents, chunk_size=500, overlap=0.2)
    print(f"generated: {len(chunked_docs)} chunks.\n")

    # Preview first chunk
    first_chunk = chunked_docs[0]
    print("Source:", first_chunk['source'])
    if first_chunk.get("page"):
        print("Page:", first_chunk['page'])
    if first_chunk.get("title"):
        print("Title:", first_chunk['title'])
    print("Content preview:", first_chunk['content'][:500], "...\n")


    ## Embedding vector
    texts = [doc["content"] for doc in chunked_docs]
    embedder = Embedder(provider="sbert", model_name="all-MiniLM-L6-v2")
    embeddings = embedder.emb(texts)

    ## Creating vector database using faiss
    print(f"embeddings: {embeddings}")
    dim = len(embeddings[0])
    store = VectorStore(dim)
    store.add(embeddings, chunked_docs)
    store.save()

    ## Query

    query = "transformer?"
    query_emb = embedder.emb([query])[0]
    results = store.search(query_emb, top_k=5)

    print("🔎 Search Results:")
    for r in results:
        print(f"- {r['content'][:200]}... (Source: {r['source']}, Score: {r['score']:.2f})")
