from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import requests  # For SerpApi requests
import getpass
import os

os.environ["GROQ_API_KEY"] = getpass.getpass()
os.environ["SERPAPI_API_KEY"] = getpass.getpass("Enter your API key: ")



file_path = input("Upload your PDF file path: ")
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} document chunks.")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

texts = [split.page_content for split in splits]
vectors = embedding_model.encode(texts, convert_to_tensor=False)

# Vector Store
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings()
)
retriever = vectorstore.as_retriever()

def search_web(query):
    params = {
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "num": 2,
    }
    response = requests.get("https://serpapi.com/search.json", params=params)
    results = response.json()
    web_content = []

    # Parse results to retrieve snippets and URLs
    for result in results.get("organic_results", []):
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        web_content.append(f"{snippet} (Source: {link})")

    return "\n".join(web_content)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved PDF content and web search results to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use concise language."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

llm = ChatGroq(model="llama3-8b-8192")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def query_agent(query):
    pdf_results = rag_chain.invoke({"input": query})
    web_results = search_web(query)

    combined_context = f"PDF Context:\n{pdf_results}\n\nWeb Search Context:\n{web_results}"

    messages = [
        SystemMessage(content="You are an assistant for answering questions based on PDF and web data."),
        SystemMessage(content=combined_context),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    answer_content = response.get("content", "I'm here to help with more questions if needed.")
    return answer_content

def query_agent(query):
    pdf_results = rag_chain.invoke({"input": query})
    web_results = search_web(query)

    combined_context = f"PDF Context:\n{pdf_results}\n\nWeb Search Context:\n{web_results}"

    messages = [
        SystemMessage(content="You are an assistant for answering questions based on PDF and web data."),
        SystemMessage(content=combined_context),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)

    answer_content = response.content if hasattr(response, "content") else "Sorry, I couldn't find an answer."
    return answer_content

while True:
    user_query = input("Enter your question (type 'exit' to stop): ")
    if user_query.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    response = query_agent(user_query)
    print("\nAnswer:", response, "\n")

