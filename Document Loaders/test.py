from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

data = TextLoader("Document Loaders/notes.txt")
docs = data.load()

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))
print(chunks[0].page_content)