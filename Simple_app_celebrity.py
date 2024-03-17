import os
import streamlit as st

from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

st.title("Simple celebrity search")
input_text = st.text_input("Give celebrity name")


first_prompt = PromptTemplate.from_template("Tell me about the celebrity {name}")
second_prompt = PromptTemplate.from_template("give me only the date of birth of {person}")
third_prompt = PromptTemplate.from_template("Mention 5 major events happened around {dob} in the world")

person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_memory")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="dob_memory")
major_memory = ConversationBufferMemory(input_key="dob", memory_key="event_memory")


llm=OpenAI(temperature=0.8)

chain = LLMChain(
    llm=llm, prompt=first_prompt, memory=person_memory, output_key="person"
)

chain_2 = LLMChain(
    llm=llm, prompt=second_prompt, memory=dob_memory, output_key="dob"
)

chain_3 = LLMChain(
    llm=llm, prompt=third_prompt, memory=major_memory, output_key="description"
)

parent_chain = SequentialChain(
    chains=[chain, chain_2, chain_3], input_variables=["name"], output_variables=["person", "dob", "description"], verbose=True
)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander("Major events"):
        st.info(major_memory.buffer)