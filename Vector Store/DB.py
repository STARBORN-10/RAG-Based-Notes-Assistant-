
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load PDF
data = PyPDFLoader("Document Loaders/RagMLDL.pdf")
docs = data.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

#print("Vector database created successfully!")
#print("Total chunks stored:", len(chunks))

result = vectorstore.similarity_search("What is the main topic of the document?", k=3)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
