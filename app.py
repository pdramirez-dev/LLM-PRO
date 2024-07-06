import os
import time
import pandas as pd
import PyPDF2
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from lib import pdf_extract

load_dotenv()

_api_key = os.getenv["OPENAI_API_KEY"]

def processTextToLLM(texto):
    model = ChatOpenAI(
        api_key=_api_key, 
        temperature=0.5, 
        model="gpt-4-turbo")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Message ",
            ),
            ("human", "{texto}"),
        ])
    chain = prompt | model
    response = chain.invoke(
        {

            "texto": texto,
        }
    )
    return response.content
