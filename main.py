from http.client import responses

from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task, description=None):
    """ Add a new task to the user's task list. Use this when the user wants to add a new task"""
    todoist.add_task(content=task,
                     description=description)

@tool
def show_task():
    """Show all the task from todoist. Use this tool when the user wants to see their tasks"""
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

tools = [add_task, show_task]
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt = """
        You are a helpful and organized task assistant.
        Your responsibilities are:
        
        1. Add new tasks when the user requests. Ask for clarification if needed.
        2. Show the user’s existing tasks in a clear, numbered or bulleted list.
        3. Confirm actions after adding tasks or showing tasks.
        4. Always keep responses concise, friendly, and easy to read.
"""



prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

#chain = prompt | llm | StrOutputParser()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# response = chain.invoke({"input": user_prompt})

history = []

while True:
    user_input= input("You: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))
