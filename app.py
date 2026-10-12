
import streamlit as st
import requests
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Basic AI Agent", page_icon="🤖")
st.title("🤖 Basic AI Agent")
st.caption("Gemini • Web Search • Weather")

@tool
def get_weather_data(city: str) -> str:
    """Get current weather for a city."""
    api_key = st.secrets.get("WEATHER_API_KEY", "")
    if not api_key:
        return "Weather API key is not configured."

    response = requests.get(
        "https://api.weatherstack.com/current",
        params={"access_key": api_key, "query": city},
        timeout=20
    )
    return str(response.json())


@st.cache_resource
def create_agent_executor():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0
    )

    tools = [DuckDuckGoSearchRun(), get_weather_data]

    template = """Answer the question using the available tools.

{tools}

Use this format:
Question: the input question
Thought: what to do
Action: one of [{tool_names}]
Action Input: the tool input
Observation: the tool result
... (repeat if needed)
Thought: I know the final answer
Final Answer: the answer

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=5
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about anything, or check weather...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                agent = create_agent_executor()
                result = agent.invoke({"input": question})
                answer = result["output"]
            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
