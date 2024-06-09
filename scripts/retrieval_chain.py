from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
# from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# docA = Document(
#     page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."
# )

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

def create_vector_db(docs):
    embedding = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(docs, embedding)
    print(vector_db)
    return vector_db

def create_chain(vector_db: FAISS):
    model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain,
    )
    
    return retrieval_chain

docs = get_documents_from_web("https://python.langchain.com/docs/expression_language/")
vector_db = create_vector_db(docs)
chain = create_chain(vector_db)

# be mindful of context size
response = chain.invoke({"input": "What are main features of LCEL?"})

print(response["context"])