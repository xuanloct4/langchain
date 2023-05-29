
import environment
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding
from agents_tools import DocstoreExplorer_lookup_tool, DocstoreExplorer_search_tool

from langchain import Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
# docstore=DocstoreExplorer(Wikipedia())
agentType = AgentType.REACT_DOCSTORE
search_tool, _ =DocstoreExplorer_search_tool()
lookup_tool, _ = DocstoreExplorer_lookup_tool()
tools = [search_tool, lookup_tool]

react = initialize_agent(tools, llm, agent=agentType, verbose=True)
question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
print(react.run(question))