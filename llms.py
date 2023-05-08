
import environment


def OpenAILLM():
    from langchain import OpenAI
    llm = OpenAI(temperature=0)
    return llm

def defaultLLM():
    from langchain.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callbacks = [StreamingStdOutCallbackHandler()]
    local_path = '../gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin' 
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    return llm