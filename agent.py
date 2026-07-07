import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from tools import execute_code, load_dataframe, web_search


load_dotenv()
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    iteration_count: int



tools = [execute_code,web_search]
llm = ChatAnthropic(model="claude-sonnet-5")
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a data analyst agent. You have access to Python
sandbox via execute_code tool, with pandas available as 'pd'. A dataframe
called 'df' is already loaded - start by inspecting it (df.head(),df.info(),
 df.describe()) before answering any question. Show your work: run code, read
 the output, and only give a final answer once you've actually verified it with 
 code, not guessed."""

def agent_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    current_count = state.get("iteration_count",0)
    return {"messages": [response], "iteration_count": current_count + 1}


tools_node = ToolNode(tools)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if state.get("iteration_count",0) >= 5:
        return END
    if getattr(last_message,"tool_calls",None):
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent",agent_node)
graph.add_node("tools",tools_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent",should_continue,{"tools":"tools",END:END})
graph.add_edge("tools","agent")

app = graph.compile()

def run_query(question: str, csv_path: str):
    load_dataframe(csv_path)
    result = app.invoke({"messages": [("user", question)], "iteration_count": 0})
    final_message = result["messages"][-1]
    if not final_message.content:
        return "(Agent hit the iteration limit without producing a final answer.)"
    return final_message.content


if __name__ == "__main__":
    answer = run_query(
        question="What are the column names, and what's the average of the first numeric column?",
        csv_path="toy_data.csv",
    )
    print("\n=== FINAL ANSWER ===")
    print(answer)
    