import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable is not set.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define State
class State(TypedDict):
    name: str
    career: str
    education: str
    year: str
    advise: str

def generate_career_advise(state: State):
    msg = llm.invoke(f"""
        You are an expert career adviser.
        Generate a detailed career plan for user whose name is {state["name"]}.
        User wants to build a career in {state['career']}.
        User's educational qualifications are: {state['education']}.
        User is currently studying in {state['year']}.
        Create the career plan in MarkDown formatting.
    """)
    return {"advise": msg.content}

# Build the graph
graph = StateGraph(State)
graph.add_node("career planner", generate_career_advise)
graph.add_edge(START, "career planner")
graph.add_edge("career planner", END)
compiled_graph = graph.compile()

# Streamlit UI
st.title("Career Planner Agent")
st.write("Enter your details to get a personalized career plan:")

with st.form("career_form"):
    name = st.text_input("Name")
    career = st.text_input("Career Goal (e.g., AI Developer)")
    education = st.text_input("Educational Qualifications (e.g., Bachelors in Technology)")
    year = st.text_input("Current Year/Status (e.g., Final Year of Computer Engineering)")
    submitted = st.form_submit_button("Get Career Plan")

if submitted:
    if not all([name, career, education, year]):
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Generating your career plan..."):
            state = compiled_graph.invoke({
                "name": name,
                "career": career,
                "education": education,
                "year": year
            })
        st.markdown(state["advise"]) 