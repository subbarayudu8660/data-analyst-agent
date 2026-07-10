# Data Analyst Agent

A tool-using, memory-persistent LangGraph agent that answers natural-language questions about a dataset by writing and executing real Python code, searching the web for external context, and generating visualizations — all with tracing, guardrails, and an automated evaluation harness.

Built as a hands-on deep dive into agentic AI engineering: every core mechanism (tool calling, state management, the ReAct loop, memory, guardrails, evaluation) is implemented from first principles on top of LangGraph's primitives, rather than relying on high-level abstractions like `create_react_agent`.

## What it does

Ask it a question like:

> "What's the average salary in our dataset, and how does that compare to the current industry average for this role?"

The agent will:
1. Write and execute real pandas code against your data
2. Search the web for current industry figures
3. Synthesize both into a coherent, cited answer
4. Remember the conversation for follow-up questions
5. Generate charts on request

All of this happens through a genuine reasoning loop — the agent decides which tools to call, in what order, and when it has enough information to answer, rather than following a fixed script.

## Architecture

```
User question
     │
     ▼
┌─────────────┐   tool call requested   ┌────────────┐
│  agent_node │ ──────────────────────► │  tool_node │
│ (calls LLM) │ ◄────────────────────── │ (executes) │
└─────────────┘   tool result returned  └────────────┘
     │
     │ no tool call needed
     ▼
Final answer
```

This is a **ReAct loop** (Reason → Act → Observe, repeat) implemented as a LangGraph `StateGraph`: two nodes, one conditional edge (route to tools or end), one fixed edge (tools always return to the agent). State is a single dict threaded through every node, merged via reducers rather than overwritten — this is what allows conversation history to accumulate correctly across iterations.

## Tools

| Tool | Purpose | Notes |
|---|---|---|
| `execute_code` | Runs Python/pandas code against the loaded dataset in a persistent sandbox | Errors are caught and returned as text, letting the agent read its own mistakes and self-correct |
| `web_search` | Searches the web for current, external information via Tavily | Docstring explicitly draws the routing boundary against `execute_code` to keep tool selection reliable |
| `create_chart` | Generates matplotlib visualizations, saved to `charts/` | Reuses the same persistent sandbox namespace as `execute_code`, so it can reference already-loaded data |

The model decides which tool(s) to call, in what order, based on the question — including calling multiple tools in a single run when a question genuinely requires it (e.g., compute a number from data, look up a comparison figure, then chart both).

## Key engineering features

**Memory** — Conversations persist within a session via a LangGraph checkpointer (`MemorySaver`), keyed by `thread_id`. Follow-up questions ("what about just the Engineering department?") correctly resolve against prior context. Memory is intentionally session-scoped: it resets when the process restarts.

**Guardrails** — A hard iteration ceiling (`MAX_ITERATIONS`) prevents runaway loops, independent of the model's own judgment about whether it's done. This is a *resource guardrail* — enforced externally by the control-flow logic, not something the model can be prompted into respecting reliably.

**Tracing** — Full observability via LangSmith. Every node execution, LLM call, and tool call is traced automatically, with no code changes required in the agent logic itself — tracing hooks live in the underlying `langchain_core` layer and activate purely from environment configuration.

**Evaluation** — A golden dataset of test questions (spanning dataset-only, web-only, and multi-tool categories) is scored two ways: a fast mechanical check (`required_facts` string matching) and an LLM-as-judge pass (a separate model call scoring correctness and completeness against a reference answer, since exact-match isn't possible for questions involving live web data). Tool usage per run is also checked against expected routing, using `app.get_state()` to inspect the checkpointer's saved history without modifying the agent's normal return path.

## Project structure

```
data-analyst-agent/
├── agent.py          # Graph definition: state, nodes, edges, the ReAct loop, memory
├── tools.py          # Tool implementations: execute_code, web_search, create_chart
├── eval_dataset.py   # Golden dataset — test questions with expected facts/tools
├── eval_runner.py    # Evaluation harness — runs the golden set, scores via judge + mechanical checks
├── toy_data.csv       # Sample dataset used for development and testing
├── charts/           # Output directory for generated visualizations
└── .env              # API keys (not committed)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install langgraph langchain-anthropic langchain-core langgraph-checkpoint pandas matplotlib tavily-python python-dotenv langsmith anthropic
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=data-analyst-agent
```

## Usage

**Interactive session:**
```bash
python agent.py
```
Runs a conversational loop — ask questions, get answers with full memory of the session, type `quit` to exit.

**Run the evaluation suite:**
```bash
python eval_runner.py
```
Runs every question in the golden dataset against the live agent, scores each response, and prints a report broken down by category (dataset-only / web-only / multi-tool) so weaknesses are diagnosable rather than hidden in one aggregate number.

## Design decisions worth noting

- **No high-level agent abstraction (`create_react_agent`, etc.)** — the graph, state schema, and routing logic are hand-built specifically to understand every mechanism rather than trust a framework default. The tradeoff was made deliberately: slower to build, but the resulting understanding transfers to any future agent architecture, including ones the framework's helpers don't support.
- **Errors are returned as tool output, not raised as exceptions** — this is what allows the agent to see its own failures and retry with corrected logic, rather than crashing the whole run.
- **The LLM-as-judge is a separate, raw API call** (not routed through the agent's own LangGraph loop) — it has no tools and no role in the conversation being evaluated, keeping evaluation cleanly decoupled from the system being evaluated.

## Known limitations / not yet built

- Self-critique / reflection step (agent doesn't yet double-check its own draft answer against tool outputs before finalizing — a real gap surfaced by a bug where a failed chart-generation call was still reported as a success)
- Graceful degradation on hitting the iteration guardrail is minimal (returns a flat message rather than an honest partial summary)
- Trajectory evaluation (analyzing *how* the agent solved a problem, not just whether the final answer was correct) is deferred to a future project with a richer multi-tool/multi-agent surface, where it becomes more meaningful
- Sandbox execution (`exec()`) has no real isolation — acceptable for local, single-user development; would need Docker-level isolation for anything handling untrusted input