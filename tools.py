import io
import contextlib
import traceback
import pandas as pd
from langchain_core.tools import tool

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
    

def load_dataframe(path: str,name: str = "df") -> str:
    """
    Load a CSV into the sandbox under a givenvariable name, before the agent start.
    """
    df = pd.read_csv(path)
    _sandbox_globals[name] = df
    return f"Loaded '{path}', into variable '{name}' with shape {df.shape}"
        