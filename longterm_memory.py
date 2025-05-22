# longterm_memory.py

# pyright: reportUnknownMemberType=false

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os

class LongTermMemory:
    def __init__(self, persist_directory: str = "longterm_memory"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vstore: Chroma = self._load_vectorstore()

    def _load_vectorstore(self) -> Chroma:
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )
        else:
            # Create an empty persistent Chroma store without inserting anything
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name="longterm-memory"
            )


    def save_fact(self, text: str):
        doc = Document(page_content=text)
        self.vstore.add_documents([doc])
        

    def query(self, query: str, k: int = 3) -> List[str]:
        results = self.vstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
