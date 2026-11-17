import os
import requests
import json
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model

load_dotenv()


groq_model = init_chat_model(model="qwen/qwen3.8-27b", model_provider= "Groq")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
tavily = TavilySearch(max_results=3,topic="general")
checkpointer = InMemorySaver()

@tool
def get_weather(location: str) -> str:
    """This tool will take a location and return weather details."""
    if not weather_api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    url = "https://api.openweathermap.org/data/2.5/weather?"

    params = {
        "q" : location,
        "appid": weather_api_key,
        "units": "metric"
    }

    response = requests.get(url=url, params=params, timeout=10)
    response.raise_for_status()
    return json.dumps(response.json())

@tool
def web_search(query: str) -> str:
    """This tool is used to do web search for current and latest news, information and other purposes."""
    return json.dumps(tavily.invoke({"query": query}))

tools = [get_weather,web_search]

llm = create_agent(model=groq_model, tools=tools, checkpointer=checkpointer, system_prompt= """You are a helpful chatbot. Use get_weather tool to answer weather realtive query, use web_search tool for all current and internet search information.""")

while True:
    thread_config = {"configurable": {"thread_id": "thread_1"}}
    messages = []
    user_input = input("User: " )
    if user_input.lower() == ["quit", "bye"]:
        print("Bye")
        exit()
        
    messages.append(HumanMessage(content= user_input))

    ai_response = llm.invoke({"messages": messages},thread_config )

    # messages.append(AIMessage(content= ai_response))

    print(ai_response["messages"][-1].content)