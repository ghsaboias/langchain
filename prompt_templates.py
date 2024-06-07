from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Instantiate model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=1.5,
)

# Prompt template
# prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}.")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return gift ideas for the following subject, as a comma separated list. Don't use a numbered list."),
        ("human", "{input}")
    ]
)

# Create LLM chain
chain = prompt | llm # output from first object will be input to second object

response = chain.invoke("girlfriend")

print(type(response.content))