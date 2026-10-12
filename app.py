
import streamlit as st
import requests
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 AI Agent")
st.caption("Powered by Groq, LangChain and Tool Calling")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY", "")

def get_weather(city: str) -> str:
    if not WEATHER_API_KEY:
        return "Weather API key is missing in Streamlit Secrets."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": WEATHER_API_KEY, "query": city},
            timeout=20
        )
        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error'].get('info', 'Unknown error')}"

        current = data.get("current", {})
        location = data.get("location", {})

        return (
            f"Weather in {location.get('name', city)}, "
            f"{location.get('country', '')}: "
            f"{current.get('temperature', 'N/A')}°C, "
            f"{current.get('weather_descriptions', ['N/A'])[0]}. "
            f"Humidity: {current.get('humidity', 'N/A')}%. "
            f"Wind speed: {current.get('wind_speed', 'N/A')} km/h."
        )
    except Exception as e:
        return f"Could not fetch weather: {e}"

search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Search the internet for information, facts and current news."
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather for a city. Input should be a city name."
    )
]

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI agent. Use the Weather tool for current weather "
        "questions and Search for current or external information. "
        "Do not invent tool results. Keep answers clear and concise."
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("AI Agent is thinking..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                answer = result.get("output", "No answer returned.")
            except Exception as e:
                answer = f"Something went wrong: {e}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


  
