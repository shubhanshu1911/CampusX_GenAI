import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-preview-05-20',
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)



#########################################

# Method 1 : using Parse
# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
#######################################

#Method 2 : Using Chain
chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
