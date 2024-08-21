from src.app.app import RagApp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

RAG = RagApp()

if __name__ == "__main__":
    RAG.main()