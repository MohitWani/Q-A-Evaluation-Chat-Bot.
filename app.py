import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Define the prompt templates for generating questions and evaluating answers
question_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a question about {topic}."
)

evaluation_prompt = PromptTemplate(
    input_variables=["question", "user_answer"],
    template=(
        "Question: {question}\n"
        "User Answer: {user_answer}\n"
        "Evaluate the accuracy and relevance of the user's answer."
    )
)

# Create the LangChain for question generation and evaluation
question_chain = LLMChain(llm=llm, prompt=question_prompt)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

st.set_page_config(layout='wide')
st.title("Generative AI Question and Answer Evaluation")

# siderbar
topic = st.sidebar.selectbox('Topic', ['Geography', 'Health', 'Sports'])


# User input for topic
# topic = st.text_input("Enter a topic:", "")

if st.sidebar.button("Generate Question"):
    if topic:
        question = question_chain.run(topic)
        st.session_state.question = question
        st.write("Question:\n", question)
    else:
        st.write("Please enter a topic to generate a question.")

if 'question' in st.session_state:
    user_answer = st.text_input("Your answer to the question:", "")

    if st.button("Evaluate Answer"):
        if user_answer:
            evaluation = evaluation_chain.run(
                {"question": st.session_state.question, "user_answer": user_answer})
            st.write("Evaluation:", evaluation)
        else:
            st.write("Please provide an answer to evaluate.")
