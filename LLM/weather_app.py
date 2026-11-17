import json
import os
from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_community.tools import tool
from dotenv import load_dotenv

from pydantic import BaseModel
import requests

load_dotenv()

llm = init_chat_model(model="openai/gpt-oss-120b", max_tokens= 200, max_retries=2, model_provider= "Groq")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

class structuredWeather(BaseModel):
    location: str
    Min_temperature: float
    Max_temperature: float
    wind: str
    pressure: str
    humidity: str


# Create Tools
@tool
def get_weather(location: str, zipcode: str) -> str:
    """This tool will take a location and return weather details."""
    if not weather_api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    url = "https://api.openweathermap.org/data/2.5/weather?"

    params = {
        "q" : location,
        "zip": zipcode,
        "appid": weather_api_key,
        "units": "metric"
    }

    response = requests.get(url=url, params=params, timeout=10)
    response.raise_for_status()
    return json.dumps(response.json())

llm_with_tools = llm.bind_tools([get_weather])
    
    
# Create a weather app
messages: list = [
    SystemMessage(content= "You are a weather assistant. Response to user query by using get_weather tool. Answer to the point and concisely.")
]

user_input = input("What is your query ?")

messages.append(HumanMessage(content= user_input))

# AI message
ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

# tool message
for tool_call in ai_message.tool_calls:
    tool_message = get_weather.invoke(tool_call)
    messages.append(tool_message)

# final_response = llm.invoke(messages)


llm_with_structured_output = llm.with_structured_output(structuredWeather)
final_structured_output = llm_with_structured_output.invoke(messages)
print(final_structured_output)