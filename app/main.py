import os
import io
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import glob

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import DataFrameLoader

GOOGLE_API_KEY = 'AIzaSyB2r1O8ufJ-zelvvOlbef3ZVxJLTWPBkOg'
context = pd.read_csv('/lobbying_with_LLM/app/tweets_data.csv')
loader = DataFrameLoader(context)
documents = loader.load()

def main():
  # Application main title
  st.markdown("<h1 style='text-align: center;'> Lobbying with Google Gemini ai </h1>", unsafe_allow_html=True)
  st.write('---')
  
  # Ask Gemini
  st.markdown("<h2 style='text-align: center;'> Ask </h2>", unsafe_allow_html=True)
  question = st.input('Ask Gemini about Senatorial Stance', use_container_width=True)
    
  # Define Prompt Template
  prompt_template = """
  what is the senatorial stance on {question}, return only the list of twitter handles of senators with : probable support, probable opposition, probable neutral position.
  return list as python dictionnary with keys: Support, Opposition, Neutral. without further explanation.
  Context: \n{context}\n
  Question: \{question}\n
  """
  
  # Create Prompt
  prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
  
  # Load QA Chain
  model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=1, api_key=GOOGLE_API_KEY)
  # Load QA Chain
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  # Get Response
  response = chain({"input_documents": documents, "question": question}, return_only_outputs=True)

  st.write(response)


if __name__ == '__main__':
  main()
