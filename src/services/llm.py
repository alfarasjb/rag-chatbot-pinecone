import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine
from src.definitions.credentials import EnvVariables

logger = logging.getLogger(__name__)


class RagChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=EnvVariables.chat_model(), temperature=0.5)
        self.embedding = OpenAIEmbeddings()

        # Connect to PGVector
        self.pgvector_url = EnvVariables.pgvector_url()  # Assuming the URL is stored in environment variables
        self.engine = create_engine(self.pgvector_url)
        self.vectorstore = PGVector(embedding_function=self.embedding, collection_name="chat_vectors",
                                    connection_string=self.pgvector_url)

        self.retriever = self.vectorstore.as_retriever()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def chat(self, user_prompt: str) -> str:
        result = self.qa({"question": user_prompt})['answer']
        logger.info(f"User Query: {user_prompt}")
        logger.info(f"Chat bot response: {result}")
        return result
