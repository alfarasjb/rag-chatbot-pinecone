import os

from dotenv import load_dotenv

load_dotenv()


class Credentials:

    @classmethod
    def openai_api_key(cls) -> str:
        return os.getenv("OPENAI_API_KEY")

    @classmethod
    def pinecone_api_key(cls) -> str:
        return os.getenv("PINECONE_API_KEY")

    @classmethod
    def pinecone_environment(cls) -> str:
        return os.getenv("PINECONE_ENVIRONMENT")


class EnvVariables:

    @classmethod
    def chat_model(cls) -> str:
        return os.getenv("CHAT_MODEL")

    @classmethod
    def pinecone_index_name(cls) -> str:
        return os.getenv("INDEX_NAME", "playground")

    @classmethod
    def pgvector_url(cls) -> str:
        return ""