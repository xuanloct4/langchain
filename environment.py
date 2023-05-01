import os

from dotenv import load_dotenv

load_dotenv() 


os.environ["LANGCHAIN_HANDLER"] = os.environ.get("LANGCHAIN_HANDLER")

## Uncomment this if using hosted setup.

# os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:4173" 

## Uncomment this if you want traces to be recorded to "my_session" instead of default.

os.environ["LANGCHAIN_SESSION"] = os.environ.get("LANGCHAIN_SESSION")

## Better to set this environment variable in the terminal
## Uncomment this if using hosted version. Replace "my_api_key" with your actual API Key.

# os.environ["LANGCHAIN_API_KEY"] = "my_api_key"  


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY