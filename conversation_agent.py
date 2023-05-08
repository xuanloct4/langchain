
import environment

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

from llms import defaultLLM as llm

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools, llm(), agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
# agent_chain.run(input="hi, i am bob")

# agent_chain.run(input="what's my name?")

# agent_chain.run("what are some good dinners to make this week, if i like thai food?")

# agent_chain.run(input="tell me the last letter in my name, and also tell me who won the world cup in 1978?")

agent_chain.run(input="whats the current temperature in pomfret?")