from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print

# creating a tool

@tool
def get_text_len(text: str) -> str:
    """
    return the number of characters in a given text
    """
    return len(text)

llm = ChatMistralAI(model = "mistral-small-2506")

# tool binding

llm_with_tool = llm.bind_tools([get_text_len])
result = llm.invoke("hello")
result2 = llm_with_tool("hello")

print(result)
print()
print()
print()
print(result2)