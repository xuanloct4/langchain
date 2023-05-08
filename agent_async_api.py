
import environment
import asyncio
import time

from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from aiohttp import ClientSession
from llms import defaultLLM as llm

questions = [
    "Who won the US Open men's final in 2019? What is his age raised to the 0.334 power?",
    "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?",
    "Who won the most recent formula 1 grand prix? What is their age raised to the 0.23 power?",
    "Who won the US Open women's final in 2019? What is her age raised to the 0.34 power?",
    "Who is Beyonce's husband? What is his age raised to the 0.19 power?"
]

tools = load_tools(["google-serper", "llm-math"], llm=llm())
agent = initialize_agent(
    tools, llm(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

def sync_tasks():
    s = time.perf_counter()
    for q in questions:
        agent.run(q)
    elapsed = time.perf_counter() - s
    print(f"Serial executed in {elapsed:0.2f} seconds.")


async def async_tasks():
    loop = asyncio.get_event_loop()
    s = time.perf_counter()
    
    # If running this outside of Jupyter, use asyncio.run or loop.run_until_complete
    tasks = [agent.arun(q) for q in questions]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - s
    print(f"Concurrent executed in {elapsed:0.2f} seconds.")
    # loop.close()

# sync_tasks()
asyncio.run(async_tasks())