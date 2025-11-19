# Q1:vectorstore ChromaDB
from langchain_community.vectorstores import Chroma
from embedder import Embedder
from splitter import DocumentSplitter
from loader import DocumentLoader

class DocumentIndexer:
    def __init__(self, data_dir: str, vectorstore_dir: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.vectorstore_dir = vectorstore_dir
        self.embedder = Embedder(model_name=embedding_model_name)
        self.splitter = DocumentSplitter()
        self.loader = DocumentLoader(data_dir=self.data_dir)
        self.vectorstore = None

    def build_index(self):
        docs = self.loader.load_documents()
        chunks = self.splitter.split(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedder.embedding_model,
            persist_directory=self.vectorstore_dir
        )
        self.vectorstore.persist()
        print("Index créé et sauvegardé.")

    def query_index(self, query: str, k: int = 5):
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embedder.embedding_model
            )
        results = self.vectorstore.similarity_search(query, k=k)
        return results