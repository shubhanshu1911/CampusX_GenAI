import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-preview-05-20',
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Pydantic Object 
class Person (BaseModel) : 
    name : str = Field(description="Name of the person")
    age : int = Field(description="Age of the person")
    city : str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

########################################################
# Method 1 : using Parse
# prompt = template.invoke({'place' : 'indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)
#################################################

#Method 2 : Using Chain
chain = template | model | parser

result = chain.invoke({'place' : 'Srilakan'})

print(result)