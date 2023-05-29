
import environment
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding
from agents_tools import search_tool_serpapi

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

def agent_chain_chat_llm():
    global memory
    global tools
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [search_tool_serpapi()]
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent_chain

def agent_chain_llm():
    global memory
    global tools
    memory = ConversationBufferMemory(memory_key="chat_history")
    tools = [search_tool_serpapi()]
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent_chain

# memory = ConversationBufferMemory(memory_key="chat_history")
# agent_chain = initialize_agent([search_tool_serpapi()], llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
memory=None
tools=None
agent_chain = agent_chain_llm()
# print(agent_chain.run(input="hi, i am bob"))
# print(agent_chain.run(input="what's my name?"))
# print(agent_chain.run("what are some good dinners to make this week, if i like thai food?"))
# print(agent_chain.run(input="tell me the last letter in my name, and also tell me who won the world cup in 1978?"))
# print(agent_chain.run(input="whats the current temperature in pomfret?"))



self_ask_with_search = initialize_agent([search_tool_serpapi("Intermediate Answer")], llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
print(self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?"))