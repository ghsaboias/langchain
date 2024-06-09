from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser, StrOutputParser)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following subject."),
            ("human", "{input}")
        ]
    )
    
    parser = StrOutputParser()
    
    chain = prompt | llm | parser

    return chain.invoke("dog")


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma separated list of values, without identifying numbers."),
            ("human", "{input}")
        ]
    )
    
    parser = CommaSeparatedListOutputParser()
    
    chain = prompt | llm | parser

    return chain.invoke("girlfriend")

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract information from the following phrase. \nFormatting instructions: {format_instructions}."),
            ("human", "{phrase}")
        ]
    )
    
    class Dish(BaseModel):
        recipe: str = Field(description="name of the recipe")
        ingredients: list = Field(description="ingredients")
    
    parser = JsonOutputParser(pydantic_object=Dish)
    
    chain = prompt | llm | parser
    
    return chain.invoke({
        "phrase": "The ingredients for a spaghetti bolognese are: spaghetti, minced beef, onion, garlic, tomato sauce, and herbs.",
        "format_instructions": parser.get_format_instructions()
    })
    

# print(type(call_string_output_parser()))
print(call_json_output_parser())