import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# LangSmith (optional – won't break if keys missing)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demo with LLaMA2")
input_text = st.text_input("What is your curiosity today?")

# Ollama LLaMA2 model
llm = Ollama(model="llama2")

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
