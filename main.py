
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyPDFLoader("Document Loaders/deep_learning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarizes the text."),
        ("human", "{data}")
    ]
)

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = template.format_prompt(
    data=chunks[0].page_content
)

result = model.invoke(prompt)

print(result.content)
