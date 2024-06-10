from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# create web base loader
loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
docs = loader.load()

# split documents returned from loader into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
split_docs = splitter.split_documents(docs)

# create vector database
embedding = OpenAIEmbeddings()
vector_db = FAISS.from_documents(docs, embedding)

# create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# instantiates search tool
search = TavilySearchResults()
# instantiates retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for information about Langchain Expression Language (LCEL)"
)
# adds tool to tools list
tools = [search, retriever_tool]

# defines agent
agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

# instantiates agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })
    
    return response["output"]

if __name__ == "__main__":
    # chat history is a list of messages that the user and the assistant have exchanged
    chat_history = []
    
    while True:
        user_input = input("Type /quit to quit.\nPrompt: ")
        
        if user_input == "/quit":
            break
        
        # invoke chain with user input and chat history
        response = process_chat(agent_executor, user_input, chat_history)
        # append the user input and the assistant response to the chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)
    