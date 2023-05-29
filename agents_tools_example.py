
import environment
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.agents import initialize_agent

def multiplierTool_example1():
    from agents_tools import multiplierTool

    tool, agentType = multiplierTool()
    mrkl = initialize_agent([tool], llm, agent=agentType, verbose=True)
    print(mrkl.run("What is 3 times 4"))

def gradio_tools_example1():
    from agents_tools import gradio_tools_StableDiffusionTool
    gradio_tools_StableDiffusionTool("Please create a photo of a dog riding a skateboard")


def gradio_tools_example2():
    from langchain.memory import ConversationBufferMemory
    from agents_tools import gradio_tools_multipleTools
    tools, agentType = gradio_tools_multipleTools()
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, memory=memory, agent=agentType, verbose=True)
    output = agent.run(input=("Please create a photo of a dog riding a skateboard "
                            "but improve my prompt prior to using an image generator."
                            "Please caption the generated image and create a video for it using the improved prompt."))
    print(output)



def human_input_tool_example():
    from agents_tools import human_input_tool
    tools, agentType = human_input_tool(llm)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=agentType,
        verbose=True,
    )
    print(agent_chain.run("I need help attributing a quote"))
    # agent_chain.run("What's my friend Erivincic's surname?")



gradio_tools_example1()
# gradio_tools_example2()


# multiplierTool_example1()
# human_input_tool_example()





