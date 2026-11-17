from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage

load_dotenv()

model = init_chat_model(model= "openai/gpt-oss-120b", model_provider="groq")

conversations: list[BaseMessage] = [
    SystemMessage(content="You are a helpful chatbot assistant, Please answer clearly and to the point.")
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Bye")
        exit()
        
    conversations.append(HumanMessage(content= user_input))
    response = model.invoke(conversations)
    conversations.append(response)
    print("AI: ", response.content)

