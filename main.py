from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader("Document Loaders/GRU_Study_Notes.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are the AI that summarizes the text"),
     ("human", "{data}")]
)

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = template.format_prompt(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)