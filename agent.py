import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from tools import execute_code, load_dataframe