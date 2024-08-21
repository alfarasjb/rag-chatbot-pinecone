import streamlit as st
import logging
from src.services.llm import RagChatBot


logger = logging.getLogger(__name__)


class Chat:
    def __init__(self):
        self.chat_model = RagChatBot()

    def set_file(self):
        st.session_state.file = None

    def chat_box(self):
        bt_col, _, _ = st.columns([1, 3, 3])
        bt_col.button(label="Back", use_container_width=True, on_click=self.set_file)
        st.title("Ask a question!")
        messages = st.container(height=400)
        user_prompt = """You are an assistant that provides information on the topics in your knowledge base. First, 
        introduce yourself and your capabilities, then discuss a short summary of your knowledge base, 
        and prompt the user to ask questions about the related topics in your knowledge base."""

        initial_message = self.chat_model.chat(user_prompt)
        logger.info(f"Initial Message: {initial_message}")

        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": initial_message}
            ]

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            messages.chat_message(message['role']).write(message['content'])

        if prompt := st.chat_input("Say something"):
            messages.chat_message("user").write(prompt)
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            bot_response = self.chat_model.chat(user_prompt=prompt)
            if bot_response:
                messages.chat_message("assistant").write(bot_response)
                bot_message = {"role": "assistant", "content": bot_response}
                st.session_state.messages.append(bot_message)
