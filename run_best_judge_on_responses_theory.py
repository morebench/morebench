import argparse
import concurrent.futures
import json
import os
import random

from tqdm import tqdm

from prompts.create_prompts_for_rubric_eval import \
    create_prompt_template_judge_model
from utils import (get_judge_response, load_existing_indices_as_set,
                   prepare_criterion_data, setup_client)

parser = argparse.ArgumentParser(description='Run judge model on generated responses (MoReBench-Theory)')
parser.add_argument("--input_file", "-i", required=True, help="Path to input JSONL file with generations")
parser.add_argument("--api_key", "-ak", required=True, help="API key for OpenRouter")
parser.add_argument("--judgement_type", "-jt", default="model_resp", 
                    choices=["model_resp", "thinking_trace"], 
                    help="Which field to judge: model_resp or thinking_trace")
parser.add_argument("--num_parallel_request", "-n", type=int, default=160, 
                    help="Number of parallel requests")
parser.add_argument("--debug", "-d", action='store_true', help='Debug with only 5 examples')
parser.add_argument("--judge_model", "-jm", default="openai/gpt-oss-120b", 
                    help="Judge model to use (default: openai/gpt-oss-120b)")
parser.add_argument("--expected_samples", "-es", type=int, default=150,
                    help="Expected number of samples in input file")
parser.add_argument("--output_dir", "-o", default=None,
                    help="Output directory (default: auto-generated from input path)")

args = parser.parse_args()

# Setup output filename
if args.output_dir:
    output_filename = os.path.join(args.output_dir, 
                                   f"{args.judgement_type}_{os.path.basename(args.input_file)}")
else:
    output_filename = args.input_file.replace("generations_theory/", 
                                              f"{args.judgement_type}_judgements_theory/")

assert args.input_file != output_filename, "Input and output filenames must be different"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

print(f"Input file: {args.input_file}")
print(f"Output file: {output_filename}")
print(f"Judge model: {args.judge_model}")
print(f"Judging: {args.judgement_type}")

# Load data
with open(args.input_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]


if args.debug:
    data = data[:5]
    print("DEBUG MODE: Processing only 5 samples")
else:
    assert len(data) == args.expected_samples, f"Expected {args.expected_samples} samples but found {len(data)}"

# Prepare criterion data
criterion_data = prepare_criterion_data(data, args.judgement_type)

# Shuffle to increase cache hit
random.seed(42)
random.shuffle(criterion_data)

# Setup client
client = setup_client('openrouter', args.api_key)


def get_judgement(idx, dp):
    """Get judgment for a single criterion"""
    reasoning_resp = dp["response"]
    rubric_criterion = dp["criterion"]
    instruction_prompt = create_prompt_template_judge_model()
    prompt = f'Reasoning Response:{reasoning_resp}\n\n{instruction_prompt}\n\nRubric Criterion:{rubric_criterion}'
    
    response, input_tokens, output_tokens = get_judge_response(client, args.judge_model, prompt)
    
    dp['idx'] = idx
    dp["judgement"] = response
    dp["judge_input_tokens"] = input_tokens
    dp["judge_output_tokens"] = output_tokens
    return dp


# Process judgments
with open(output_filename, "a+") as fw:
    fw.seek(0)
    existing_idx = load_existing_indices_as_set(output_filename)
    print(f"Found {len(existing_idx)} existing rows out of {len(criterion_data)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_parallel_request) as executor:
        futures = [
            executor.submit(get_judgement, idx, dp) 
            for idx, dp in enumerate(criterion_data) 
            if idx not in existing_idx
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                fw.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Error: {e}")

print(f"Completed! Results written to: {output_filename}")