#TODO: implementer loader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

class DocumentLoader:
    def __init__(self, data_dir: str, file_type: str = "pdf"):
        self.data_dir = data_dir
        self.file_type = file_type

    def load_documents(self):
        if self.file_type == "pdf":
            loader = DirectoryLoader(self.data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        else:
            raise NotImplementedError(f"Loader pour {self.file_type} non implémenté")
        documents = loader.load()
        return documents
