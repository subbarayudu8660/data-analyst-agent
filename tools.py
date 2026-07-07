import io
import contextlib
import traceback
import pandas as pd
from langchain_core.tools import tool
from tavily import TavilyClient

_sandbox_globals = {"pd": pd}

@tool
def execute_code(code: str)->str:
    """
    Execute Python code in a persistent sandbox that has pandas available as 'pd' 
    and any dataframes already loaded. Use this to inspect data, run computations,
    and test hypotheses. The sandbox keeos state between calls, so variables you 
    define persist. Print anything you want to see in the output.
    """

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, _sandbox_globals)
        output = stdout_capture.getvalue()
        if not output.strip():
            output = "Code ran with no printed output - did you forget a print()?"
        return output
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}"
    
_tavily_client = TavilyClient()

@tool
def web_search(query: str)->str:
    """
    Search th web for current, external information -news, industry benchmarks, 
    general facts, or anything not conatined in the local dataset. Do NOT use this 
    to answer questions answerable from the dataframe; Use execute_code for that. 
    Returns a few relavant results with source and summary.
    """

    response = _tavily_client.search(query=query, max_results=3)
    results = response.get("results",[])
    if not results:
        return "No results found."
    
    formatted_results = []
    for result in results:
        formatted_results.append(f"Source: {result['url']}\n{result['content']}")
    return "\n\n".join(formatted_results)

def load_dataframe(path: str,name: str = "df") -> str:
    """
    Load a CSV into the sandbox under a givenvariable name, before the agent start.
    """
    df = pd.read_csv(path)
    _sandbox_globals[name] = df
    return f"Loaded '{path}', into variable '{name}' with shape {df.shape}"
        