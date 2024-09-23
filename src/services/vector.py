import logging
from typing import List

from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from src.definitions.credentials import Credentials, EnvVariables

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

        # PGVector configuration
        self.pgvector_url = EnvVariables.pgvector_url()
        self.collection_name = "chat_vectors"
        self.engine = create_engine(self.pgvector_url)
        self.store = PGVector(embedding_function=self.embeddings, collection_name=self.collection_name,
                              connection_string=self.pgvector_url)

    def store_to_pgvector(self, text: str):
        logger.info(f"Storing documents to PGVector Database...")
        # Clear existing vectors first
        self.clear_index()
        documents = self.get_documents(text)

        logger.info(f"Storing {len(documents)} documents...")
        self.store.add_documents(documents)

    def get_documents(self, project_string: str) -> List[Document]:
        logger.info(f"Splitting documents...")
        documents = self.text_splitter.create_documents(texts=[project_string])
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Num Splits: {len(splits)}")
        return splits

    def clear_index(self):
        logger.info(f"Clearing PGVector index...")
        try:
            self.store.delete_collection()
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
