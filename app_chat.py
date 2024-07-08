#Import required libraries

import os

import streamlit as st
import pandas as pd
import tabulate


from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

#start the terminal code >>  streamlit run D:\DADOS\Desktop\project\data_chat\app_EDA.py

#git init
#git status
#git add .
#git commit -m "commit"
#git remote add origin https://github.com/LucasZonfrilli/project.git
#git branch -M main
#git push -u origin main
#git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data_chat/apikey.py" --prune-empty --tag-name-filter cat -- --all


#from apikey import apikey


#OpenAiKey
os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]
#os.environ['OPENAI_API_KEY'] = apikey
#load_dotenv(find_dotenv())

#Welcome message
st.title('Assistente de IA para Análise de Dados 👨‍🌾')
st.write('Olá, 👋 sou seu Acessor de IA e eu estou aqui para te ajudar com análises de dados')

#Explanation sidebar
with st.sidebar:
    st.write('*Sua aventura em ciência de dados começa com um arquivo CSV.*')
    st.caption('''**Carregue seu arquivo CSV para que possamos entender e explorar seus dados.
             Depois, transformaremos seu desafio de negócios em um framework de ciência de dados e usaremos modelos de aprendizado de máquina para resolver seu problema.
             Parece divertido, certo?**
               ''')
    
    st.divider()

    st.caption("<p style ='text-align:center'> Feito com entusiasmo por Lucas Z 😁</p>", unsafe_allow_html=True)


#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Vamos começar!", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader('Carregue seu documento aqui', type='csv')
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, header=0, delimiter=';', low_memory=False)

        #llm model
        llm = OpenAI(temperature = 0)

        #Function sidebar
        @st.cache_resource
        def steps_eda():
            steps_eda = llm('Quais são os passos da Análise Exploratória de Dados')
            return steps_eda

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True, allow_dangerous_code=True)


        #Functions main
        @st.cache_resource
        def function_agent():
            st.write("**Visão Geral dos Dados**")
            st.write("The first rows of your dataset look like this?")
            st.write(df.head())
            st.write("**Limpeza de Dados**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and")
            st.write(duplicates)
            st.write("**Resumo dos Dados**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations bet")
            st.write(correlation_analysis)
            new_features = pandas_agent.run("What new features would be interess in variable")
            st.write(new_features)
            return

        @st.cache_resource
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of user {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Assess the presence of outliers the {user_question_variable}")
            st.write(normality)
            trends = pandas_agent.run(f"Analyse trends, seasonality the {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_resource
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
            

        #Main

        st.header('Análise Exploratória de Dados')
        st.subheader('Informações Gerais sobre seus Dados')

        with st.sidebar:
            with st.expander('Quais são os passos da Análise Exploratória de Dados'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Explore as Variável do Conjunto de Dados')
        user_question_variable = st.text_input('Qual variável te interessa?')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Futuros estudos')

        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else you would like to dataframe")
            if user_question_dataframe is not None and user_question_variable not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no","No"):
                st.write("")