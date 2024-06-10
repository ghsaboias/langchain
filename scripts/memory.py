import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.upstash_redis import \
    UpstashRedisChatMessageHistory
from langchain_openai import ChatOpenAI

# chat history is stored in Upstash Redis
history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
    session_id="chat1",
    # time to live in seconds (0 = never expire)
    ttl=0
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

msg1 = {
    "input": "My name is Gui",
}

response1 = chain.invoke(msg1)

print(response1)

msg2 = {
    "input": "What is my name?",
}

response2 = chain.invoke(msg2)

print(response2)