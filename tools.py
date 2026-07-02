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
        