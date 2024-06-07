from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1, # higher = more creative
    max_tokens=300,
    verbose=True,
)

# response = llm.invoke("What's the difference between green and blue?")

response = llm.stream("What's the difference between green and blue?")
for chunk in response:
    print(chunk.content, end="", flush=True)
    
# response = llm.batch(["What's the difference between green and blue?", "Write poem about AI."])
# for item in response:
#     print("Response:", item.content, "\n")
