from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import os

class RetrieverTool:
    """
    A tool for performing semantic document retrieval using Chroma and Hugging Face embeddings.

    This class supports:
    - Loading an existing Chroma vector store from disk
    - Querying the store for the most relevant documents based on a user input
    - Creating a new vector store from raw text documents via a static method

    It is designed to be used as a retrieval component within an LLM-based agent system,
    enabling context-aware responses based on a preloaded knowledge base.
    """
    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vstore = self._load_vectorstore()

    def _load_vectorstore(self):
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )
        else:
            raise ValueError("Vector store not found. You need to index documents first.")
        

    def query_v1(self, query: str, k: int = 4) -> list[str]:
        """
        [DEPRECATED] Queries the vector store for the top-k most similar documents to the input query string.

        Args:
            query (str): The query string to search for similar documents.
            k (int, optional): The number of top similar documents to retrieve. Defaults to 4.

        Returns:
            list[str]: A list containing the page content of the top-k similar documents.

        Deprecated:
            This method is deprecated and may be removed in future versions. Use the updated query method instead.
        """
        warnings.warn(
            "query_v1() is deprecated and will be removed in a future version. Use query() instead.",
            DeprecationWarning,
            stacklevel=2  # shows the warning at the caller level
        )
        docs = self.vstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def query(self, query: str, k: int = 4) -> list[str]:
        """FOR PROJECT ONLY, PRODUCTION WOULD REQUIRE USERS TO ENTER INFORMATION ABOUT DOCUMENTS THEY WILL USE RAG WITH"""
        if "bristol" not in query.lower():
            query = f"In the context of the July 4th celebrations in Bristol, Rhode Island: {query}"
        docs = self.vstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    
    @staticmethod
    def build_from_documents(
        doc_texts: list[str],
        persist_directory: str = "vectorstore"
    ):
        """
        Builds and saves a Chroma vector store from a list of raw text documents.

        This method:
        - Converts each text string into a LangChain Document object
        - Uses a SentenceTransformer embedding model to generate vector representations
        - Creates a Chroma vector store with those embeddings
        - Persists the index to the specified directory on disk

        Args:
            doc_texts (list[str]): A list of document strings to be indexed.
            persist_directory (str): The folder path where the Chroma index will be saved.

        Returns:
            None
        """
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        documents: list[Document] = [Document(page_content=txt) for txt in doc_texts]
        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        print(f"Vector store saved to `{persist_directory}`.")
