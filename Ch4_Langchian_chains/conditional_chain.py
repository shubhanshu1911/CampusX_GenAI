import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

parser1 = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-preview-05-20',
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

class feedback(BaseModel) : 
    sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classification_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Conditional Chain Syntax
# branch_chain = RunnableBranch(
#     (condition1 , chain1),
#     (condition2, chain2),
#     default chain
# )

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive' , prompt2 | model| parser1),
    (lambda x : x.sentiment == 'negative' , prompt3 | model| parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classification_chain | branch_chain

result = chain.invoke({'feedback': 'This is a beautiful phone'})

print(result)

chain.get_graph().print_ascii()
