import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-preview-05-20',
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

chat_history = [
    SystemMessage(content='You are a helper AI Assistant')
]

while True :
    user_input = input("You: ") 
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: " , result.content)

print(chat_history)