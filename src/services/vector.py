import logging
from typing import List

import pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.definitions.credentials import Credentials, EnvVariables

logger = logging.getLogger(__name__)


# TODO ADD LOGGING
class VectorDatabase:
    def __init__(self):
        self.index_name = EnvVariables.pinecone_index_name()
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        self.store = PineconeVectorStore.from_existing_index(embedding=self.embeddings, index_name=self.index_name)

    def store_to_pinecone(self, text: str):
        logger.info(f"Storing documents to Vector Database..")
        # Delete all vectors first
        self.clear_index()
        documents = self.get_documents(text)
        PineconeVectorStore.from_documents(documents, self.embeddings, index_name=self.index_name)

    def get_documents(self, project_string: str) -> List[Document]:
        logger.info(f"Splitting documents..")
        documents = self.text_splitter.create_documents(texts=[project_string])
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Num Splits: {len(splits)}")
        return splits

    def clear_index(self):
        logger.info(f"Deleting all vectors from Pinecone index.")
        pc = pinecone.Pinecone(
            api_key=Credentials.pinecone_api_key(),
            environment=Credentials.pinecone_environment()
        )

        index = pc.Index(self.index_name)
        try:
            index.delete(delete_all=True)
        except Exception as e:
            logger.info(f"Nothing to delete..")