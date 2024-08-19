import streamlit as st

from src.app.chat import Chat


class RagApp:
    def __init__(self):
        self.chat = Chat()

    def main(self):
        self.chat.chat_box()
