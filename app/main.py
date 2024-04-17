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
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
context = pd.read_csv('app/tweets_data.csv')
loader = DataFrameLoader(context)
documents = loader.load()

def get_sen_stance(documents, question):
  '''
  generate answer from Gemini ai
  '''
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
  model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=1, api_key=os.environ['GOOGLE_API_KEY'])
  # Load QA Chain
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  # Get Response
  response = chain({"input_documents": documents, "question": question}, return_only_outputs=True)

  return response

def dataframe_answer(response, tweet_data_link='app/tweets_data.csv'):
  '''
  Tranforme the response provided by Gemini AI into a readable dataframe of
  Senators who Support, Oppose, Neutral or Undecided vis-a-vis the question asked
  '''
  # cleaning dicitonnary
  for a, b in response.items():
    answer = eval(b.strip('```python\n'))

  # Initial DataFrame
  initial_df = pd.DataFrame.from_dict(answer, orient='index').T

  # generate a list of all senators in response
  senat = [j for i in answer.values() for j in i]

  # filter undecided senators
  tweet_data = pd.read_csv(tweet_data_link)
  undecided = ['@'+i for i in tweet_data['Twitter_handles'].values if i not in senat]
  undecided_df = pd.DataFrame(undecided, columns=['Undecided'])

  # generate and format final dataframe
  df_final = pd.concat([initial_df, undecided_df], axis=1)
  df_final.fillna('', inplace=True)

  # return result dataframe
  return df_final, answer
  

def main():
  # Application main title
  st.markdown("<h1 style='text-align: center;'> Lobby{ai}st Buddy </h1>", unsafe_allow_html=True)
  st.write('---')

  # with st.sidebar:
    # st.text_input('Insert Your API Key', GOOGLE_API_KEY, type='password')
  
  # Ask Gemini
  st.markdown("<h2 style='text-align: center;'> Ask </h2>", unsafe_allow_html=True)
  question = st.text_input('Ask Gemini-AI about Senatorial Stance', 'debt ceiling')
  #Button for submit
  ask = st.button('Analyze!', use_container_width=True)

  #When button is clicked
  if ask:
    response = get_sen_stance(documents, question)
    st.write('---')
      
    # Results
    st.markdown("<h2 style='text-align: center;'> Senators Lists </h2>", unsafe_allow_html=True)
    df, answer = dataframe_answer(response, 'app/tweets_data.csv')
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.write('---')
  
    st.markdown("<h2 style='text-align: center;'> PieChart </h2>", unsafe_allow_html=True)
    # PieChart of Senatorial Stance    
    # compute the size of remaining undecided senators
    len_undecided = 100 - sum([len(i) for i in answer.values()])
    labels = [i for i in answer.keys()] + ['Undecided']
    sizes = [len(i) for i in answer.values()] + [len_undecided]
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%.1f%%')
    st.pyplot(fig)
    st.write('---')

if __name__ == '__main__':
  main()
