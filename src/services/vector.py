from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import pinecone
from tqdm import tqdm

from src.definitions.credentials import Credentials


# TODO ADD LOGGING
class VectorDatabase:
    def __init__(self):
        self.index_name = "playground"
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        self.store = PineconeVectorStore.from_existing_index(embedding=self.embeddings, index_name=self.index_name)

    def store_to_pinecone(self, text: str):
        # Delete all vectors first
        self.clear_index()
        documents = self.get_documents(text)
        # self.store.add_documents(documents)
        PineconeVectorStore.from_documents(documents, self.embeddings, index_name=self.index_name)
        # vectorstore = PineconeVectorStore.from_documents(documents, self.embeddings, index_name=self.index_name)
        # print(vectorstore)

    def get_documents(self, project_string: str):
        documents = self.text_splitter.create_documents(texts=[project_string])
        splits = self.text_splitter.split_documents(documents)
        print(f"Splits: {len(splits)}")
        return splits

    def clear_index(self):
        print(f"Deleting all vectors from pinecone index.")
        pc = pinecone.Pinecone(
            api_key=Credentials.pinecone_api_key(),
            environment=Credentials.pinecone_environment()
        )

        index = pc.Index(self.index_name)
        try:
            index.delete(delete_all=True)
        except Exception as e:
            print(f"Nothing to delete..")