from tools.retriever import RetrieverTool
import os

from typing import List

def load_text_files(folder: str = "data/documents") -> List[str]:
    texts: List[str] = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        with open(path, "r") as f:
            texts.append(f.read())
    return texts

texts = load_text_files()
RetrieverTool.build_from_documents(texts)
