import argparse
from utils import (
    load_judgement_data,
    group_criteria_by_task,
    calculate_task_scores,
    calculate_all_metrics,
    format_results_row
)

parser = argparse.ArgumentParser(description='Calculate MoReBench scores from judgement data')
parser.add_argument("--input_file", "-i", required=True, help="Path to judgement JSONL file")
parser.add_argument("--format", "-f", default="latex", choices=["latex", "human"], 
                    help="Output format: latex (table row) or human (readable)")
parser.add_argument("--expected_samples", "-es", type=int, default=11568,
                    help="Expected number of judgement entries (default: 11568)")

args = parser.parse_args()

# Load and validate data
data = load_judgement_data(args.input_file)
assert len(data) == args.expected_samples, \
    f"Expected {args.expected_samples} entries, got {len(data)}"

# Group criteria by task
task_id_to_criteria = group_criteria_by_task(data)

# Calculate scores
task_id_to_score = calculate_task_scores(task_id_to_criteria)

# Define analysis categories
TASK_CATEGORIES = [None, "dilemma_source", "role_domain", "dilemma_type"]
CRITERION_CATEGORIES = ["criterion_dimension", "criterion_weight"]
TOKEN_FIELDS = ["input_tokens", "output_tokens", "len"]

# Calculate all metrics
all_results = calculate_all_metrics(
    data=data,
    task_id_to_criteria=task_id_to_criteria,
    task_id_to_score=task_id_to_score,
    task_categories=TASK_CATEGORIES,
    criterion_categories=CRITERION_CATEGORIES,
    token_fields=TOKEN_FIELDS,
    human_readable=(args.format == "human")
)

# Output formatted results
RESULT_FIELDS = [
    'daily_dilemmas', 'ai_risk', 'expert_case', 'short_case', 'long_case',
    'ai_advisor', 'ai_agent', "overall", "len"
]

if args.format == "latex":
    print(format_results_row(all_results, RESULT_FIELDS))
else:
    # Human-readable format already printed by calculate_all_metrics
    print("\n=== Summary ===")
    print(f"Overall Score: {all_results['overall']}")
    print(f"Average Response Length: {all_results['len']} chars")
    if 'overall' in all_results and 'len' in all_results:
        normalized = round(all_results["overall"] / all_results["len"] * 1000, 1)
        print(f"Normalized Score (per 1k chars): {normalized}")