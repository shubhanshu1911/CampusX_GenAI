from langchain_core.prompts import ChatPromptTemplate

Chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple term, what is {topic}')
]) 

prompt = Chat_template.invoke({'domain' : 'Cricket', 'topic' : 'Dusra'})

print(prompt)