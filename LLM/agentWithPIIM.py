import os
import json
import requests
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import SummarizationMiddleware, ModelFallbackMiddleware
from langchain_core.runnables import RunnableConfig
from groq import RateLimitError as GroqRateLimitError

load_dotenv()


# Model defination
llm_primary = init_chat_model(model="qwen/qwen3.8-27b",model_provider="Groq",profile={"max_input_tokens": 131072})
llm_fallback_1 = init_chat_model(model="openai/gpt-oss-120b",model_provider="Groq",profile={"max_input_tokens": 131072})
llm_fallback_2 = init_chat_model(model="openai/gpt-oss-20b",model_provider="Groq",profile={"max_input_tokens": 131072})
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
tavily = TavilySearch(max_results=3,topic="general")

# Tools
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

llm = create_agent(
    model= llm_primary, 
    tools=tools, 
    checkpointer=InMemorySaver(), 
    middleware=[SummarizationMiddleware(model=llm_fallback_1, trigger=("fraction", 0.7), keep=("fraction", 0.2)), ModelFallbackMiddleware(llm_fallback_1, llm_fallback_2)],
    system_prompt= """You are a helpful chatbot. Use get_weather tool to answer weather realtive query, use web_search tool for all current and internet search information.""")

thread_config: RunnableConfig = {"configurable": {"thread_id": "thread_1"}}
while True:
    messages = []
    user_input = input("User: " )
    if user_input.lower() == ["quit", "bye"]:
        print("Bye")
        exit()
        
    messages.append(HumanMessage(content= user_input))

    try:
        ai_response = llm.invoke({"messages": messages}, config=thread_config)

        print(ai_response["messages"][-1].content)
        print(ai_response["messages"][-1].usage_metadata)

    except GroqRateLimitError as error:
        print(f"Rate Limit Exception : {error}")

    except Exception as e:
        print(f"Error details: {e}")