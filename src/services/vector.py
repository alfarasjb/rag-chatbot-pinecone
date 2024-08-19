from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

from src.definitions.credentials import Credentials


# TODO
class Database:
    def __init__(self):
        self.index_name = "playground"
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.store = PineconeVectorStore.from_existing_index(embedding=self.embeddings, index_name=self.index_name)

    def store_to_pinecone(self, documents):
        for document in tqdm(documents, desc="Storing documents to Vector Database"):
            project = ""
            project_string = self.project_to_string(project)
            documents = self.get_documents(project_string)
            vectorstore = PineconeVectorStore.from_documents(documents, self.embeddings, index_name=self.index_name)

    def get_documents(self, project_string: str):
        documents = self.text_splitter.create_documents(texts=[project_string])
        return self.text_splitter.split_documents(documents=documents)

    def project_to_string(self, project: Dict[str, Any]):
        project_string = ""
        for key, value in project.items():
            project_string += f'{key}: {value}\n'
        return project_string