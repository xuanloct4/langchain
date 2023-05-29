

import environment
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI


from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

# Define your embedding model
# embedding = OpenAIEmbeddings()

# Initialize the vectorstore as empty
import faiss

# embedding_size = 1536   #For chatgpt OpenAI
embedding_size = 768      #For HuggingFace
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embedding.embed_query, index, InMemoryDocstore({}), {})

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import SerpAPIWrapper, LLMChain

todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=llm, prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)



OBJECTIVE = "Write a weather report for SF today"

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})

