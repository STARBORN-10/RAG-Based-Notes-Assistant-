from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("Document Loaders/GRU_Study_Notes.pdf")

docs = data.load()

print(docs[2])