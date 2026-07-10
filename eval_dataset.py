GOLDEN_DATASET = [
    # --- Dataset-only questions (execute_code) ---
    {
        "id": "data_1",
        "question": "What is the average salary across all employees?",
        "expected_answer": "85500",
        "required_facts": ["85500", "85,500"],
        "expected_tools": ["execute_code"],
        "category": "dataset_only"
    },
    {
        "id": "data_2",
        "question": "Who has the highest salary and what department are they in?",
        "expected_answer": "Carol, Engineering, $110,000",
        "required_facts": ["Carol", "Engineering", "110000", "110,000"],
        "expected_tools": ["execute_code"],
        "category": "dataset_only"
    },
    {
        "id": "data_3",
        "question": "What is the average years of experience in the Sales department?",
        "expected_answer": "5.5 years",
        "required_facts": ["5.5"],
        "expected_tools": ["execute_code"],
        "category": "dataset_only"
    },
    {
        "id": "data_4",
        "question": "How many employees have a salary above $90,000?",
        "expected_answer": "2 (Alice at $95,000 and Carol at $110,000)",
        "required_facts": ["2", "Alice", "Carol"],
        "expected_tools": ["execute_code"],
        "category": "dataset_only"
    },
    # --- Web-only questions (web_search) ---
    {
        "id": "web_1",
        "question": "What is the current median salary for software engineers in the United States according to recent data?",
        "expected_answer": None,  # can't know exact number, judge checks reasonableness
        "required_facts": [],  # LLM-as-judge will assess
        "expected_tools": ["web_search"],
        "category": "web_only"
    },
    {
        "id": "web_2",
        "question": "What is the projected average base pay increase percentage for US workers in 2026?",
        "expected_answer": None,
        "required_facts": [],
        "expected_tools": ["web_search"],
        "category": "web_only"
    },
    # --- Multi-tool questions (both) ---
    {
        "id": "multi_1",
        "question": "What is the average Engineering salary in our dataset, and how does it compare to the national average for software engineers in the US?",
        "expected_answer": None,
        "required_facts": ["97667", "97,667"],  # dataset fact must be present
        "expected_tools": ["execute_code", "web_search"],
        "category": "multi_tool"
    },
    {
        "id": "multi_2",
        "question": "Which department in our dataset has the lowest average salary, and is that figure above or below the US median household income?",
        "expected_answer": None,
        "required_facts": ["Marketing", "68000", "68,000"],
        "expected_tools": ["execute_code", "web_search"],
        "category": "multi_tool"
    },
]