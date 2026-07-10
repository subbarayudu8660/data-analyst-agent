import json
from anthropic import Anthropic
from dotenv import load_dotenv

from agent import run_query,app
from eval_dataset import GOLDEN_DATASET

load_dotenv()
judge_client = Anthropic()

JUDGE_SYSTEM_PROMPT = """
You are an evaluator scoring an AI agent's answer to a question

You will be given:
-The original question
- An expected answer or reference facts (may be approximate, especially for questions about current/live data)
- The AI agent's actual answer

Score the agent's answer on a scale of 1 to 5, where:
5 = Fully correct and complete
4 = correct but missing minor detail or slightly unclear
3 = Partially correct, but some real issues
2 = Mostly incorrect, but shows some right direction
1 = Completely incorrect or irrlelevant

Respond only with valid JSON in this exact format, nothing else:
{"score": <int 1-5>, "reasoning" :"<one sentence explanation>"}
"""

def judge_answer(question: str, expected_answer: str, actual_answer: str) -> dict:
    expected_text = expected_answer if expected_answer else "(No fixed expected answer - this question involves current/live data. Judge whether the answer is reasonable, well-reasoned and appropriately sourced.)"

    userprompt = f"""Question: {question}
    Expected_answer: {expected_text}
    Actual_answer: {actual_answer}
    Score this answer."""

    response = judge_client.messages.create(
        model= "claude-sonnet-5",
        max_tokens = 300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role":"user","content" : userprompt}]
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except:
        return {"score" : None , "reasoning" : f"Failed to parse judge response: {raw}"}
    
def check_required_facts(answer: str, required_facts: list)-> bool:
    if not required_facts:
        return None
    return any(fact.lower() in answer.lower() for fact in required_facts)

def get_tools_used(thread_id: str)-> list:
    state = app.get_state({"configurable" : {"thread_id":thread_id}})
    tools = []
    for msg in state.values["messages"]:
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tools.append(tc["name"])
    return tools

def run_eval():
    results = []

    for item in GOLDEN_DATASET:
        print(f"Running {item['id']} : {item['question'][:60]}...")
    
        thread_id = f"eval-{item["id"]}"
        answer = run_query(item["question"],"toy_data.csv",thread_id = thread_id)
        facts_pass = check_required_facts(answer,item["required_facts"])
        judge_result = judge_answer(item['question'],item["expected_answer"],answer)
        tools_used = get_tools_used(thread_id)
        tools_match = set(tools_used) == set(item["expected_tools"])

        results.append({
            "id" : item["id"],
            "category" : item["category"],
            "question" : item["question"],
            "answer" : answer,
            "required_facts_pass" : facts_pass,
            "judge_score" : judge_result["score"],
            "judge_reasoning" : judge_result["reasoning"],
            "expected_tools": item["expected_tools"],
            "tools_used" : tools_used,
            "tools_match" : tools_match
        })

    print_report(results)
    return results

def print_report(results):
    print("\n" + "=" * 60)
    print("EVAL REPORT")
    print("=" * 60)

    for r in results:
        print(f"\n[{r['id']}] ({r['category']})")
        print(f"  Judge score: {r['judge_score']}/5 — {r['judge_reasoning']}")
        if r["required_facts_pass"] is not None:
            print(f"  Required facts: {'PASS' if r['required_facts_pass'] else 'FAIL'}")
        print(f"  Tools used: {r['tools_used']} (expected: {r['expected_tools']}) — {'MATCH' if r['tools_match'] else 'MISMATCH'}")

    scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else 0
    tool_match_rate = sum(1 for r in results if r["tools_match"]) / len(results)

    print("\n" + "-" * 60)
    print(f"Average judge score: {avg_score:.2f}/5")
    print(f"Tool routing match rate: {tool_match_rate:.0%}")

    for category in ["dataset_only", "web_only", "multi_tool"]:
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            cat_scores = [r["judge_score"] for r in cat_results if r["judge_score"] is not None]
            cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            print(f"  {category}: {cat_avg:.2f}/5 avg ({len(cat_results)} questions)")


if __name__ == "__main__":
    run_eval()
