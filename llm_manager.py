from dotenv import load_dotenv
load_dotenv() ## activating up all the secret variables 
from validation import PropertyQuery
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "gpt-4.1")
structured_llm = llm.with_structured_output(PropertyQuery)

class LLM:
    def __init__(self):
        self.llm = llm
        self.structured_llm = structured_llm

    def get_llm(self):
        return self.llm
    def get_structured_llm(self):
        return self.structured_llm

