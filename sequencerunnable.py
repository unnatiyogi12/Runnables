from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

#  these are the runnables
"""

runnables help to make all things just by invoke message 
model created using runnables directly
that is we can make promplt using invoke , parse parses using invoke 
directly connected through the chain which follows the order

"""
chain = prompt | model | parser

result = chain.invoke("Machine Learning")
print(result)