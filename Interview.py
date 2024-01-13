import sys
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = []
for file_name in os.listdir("./docs"):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(os.path.join(os.getcwd(), "docs"), file_name)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents, embedding=OpenAIEmbeddings(), persist_directory="./data"
)
vectordb.persist()

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(
    f"{yellow}---------------------------------------------------------------------------------\n",
    "Welcome to Nika.eco. You are now ready to start interacting with your documents\n",
    "To quit, type 'exit', 'quit', 'q'.\n",
    "---------------------------------------------------------------------------------",
)
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q":
        print("Exiting")
        sys.exit()
    if query == "":
        print("You have input blank, please input a prompt.")
        continue
    result = pdf_qa.invoke({"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))
