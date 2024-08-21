import streamlit as st

from src.app.chat import Chat
from PyPDF2 import PdfReader
from src.services.vector import VectorDatabase

class RagApp:
    def __init__(self):
        self.chat = Chat()
        self.vector_db = VectorDatabase()
        self.initialize_session_state()

    def initialize_session_state(self):
        if "file" not in st.session_state:
            st.session_state.file = None

    def file_uploader(self):
        st.title(f"Upload a file..")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=False)
        text = ""
        if uploaded_file is not None:
            # read the file
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text()
            st.session_state.file = text
            st.button(label="Upload to Vector Database", use_container_width=True, on_click=self.upload_to_vector_db, args=(text,))
        st.header("Or use the current contents..")
        st.button(label="Start chatting!", use_container_width=True, on_click=self.set_file_value)

    def set_file_value(self):
        st.session_state.file = "File"

    def upload_to_vector_db(self, texts: str):
        print(f"Uploading to vector db")
        if "messages" in st.session_state:
            del st.session_state['messages']
        # self.vector_db.clear_index()
        self.vector_db.store_to_pinecone(texts)

    def main(self):
        print(f"File: {st.session_state.file}")
        if st.session_state.file is None:
            self.file_uploader()
        else:
            self.chat.chat_box()
