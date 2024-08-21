import logging

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.definitions.credentials import Credentials, EnvVariables

logger = logging.getLogger(__name__)


class RagChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=EnvVariables.chat_model(), temperature=0.5)
        self.embedding = OpenAIEmbeddings()
        self.index_name = EnvVariables.pinecone_index_name()
        self.vectorstore = PineconeVectorStore(embedding=self.embedding, index_name=self.index_name)
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
