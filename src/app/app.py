import streamlit as st

from src.app.chat import Chat
from PyPDF2 import PdfReader
from src.services.vector import VectorDatabase

class RagApp:
    def __init__(self):
        self.chat = Chat()
        self.vector_db = VectorDatabase()

    def file_uploader(self):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=False)
        text = ""
        if uploaded_file is not None:
            # read the file
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text()
            st.text_area("PDF Content: ", text, height=300)

            if st.button(label="Upload to Vector Database", use_container_width=True):
                if text == "":
                    return
                self.upload_to_vector_db(text)

    def upload_to_vector_db(self, texts: str):
        print(f"Uploading to vector db")
        self.vector_db.store_to_pinecone(texts)

    def main(self):
        self.file_uploader()
        self.chat.chat_box()
