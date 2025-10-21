import argparse
import concurrent.futures
import os
from ast import literal_eval

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from prompts.create_prompts_for_reasoning_eval import \
    create_prompt_template_for_reasoning_eval_natural_behavior
from utils import (collect_response, collect_thinking_response,
                   get_model_filename, load_existing_indices, setup_client,
                   write_to_jsonl)

parser = argparse.ArgumentParser(description='run inferences on MoReBench dilemmas')
parser.add_argument("--api_provider", "-ap", required=True, choices=['openai','anthropic','togetherai','xai','openrouter'])
parser.add_argument("--api_key", "-ak", required=True, help="API key for the service.")
parser.add_argument("--model", "-m", required=True)
parser.add_argument("--num_parallel_request", "-n", type=int, default=100)
parser.add_argument("--generations_dir", "-g", default="generations", required=False)
parser.add_argument("--debug", "-d", action='store_true', help='debug with only 5 examples')
parser.add_argument("--reasoning", "-r", action='store_true', help='use reasoning/thinking mode if available')
parser.add_argument("--budget_tokens", "-b", type=int, default=10000, help='budget tokens for thinking/reasoning mode')
parser.add_argument("--input_file", "-i", default="dataset_11092025.csv", help="Path to the input CSV file")
parser.add_argument("--reasoning_effort", "-re", default="medium", choices=['minimal','low', 'medium', 'high'])
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--hf_token","-ht", required=True)

args = parser.parse_args()

# Setup
os.makedirs(args.generations_dir, exist_ok=True)
client = setup_client(args.api_provider, args.api_key)
model_for_filename = get_model_filename(args.model, args.api_provider)
output_file_jsonl = f'{args.generations_dir}/{model_for_filename}_reasoning_{args.reasoning_effort}_seed_{args.seed}.jsonl'
print(f"Output file: {output_file_jsonl}")


def process_single_row(row, idx): 
    new_row = row.copy()
    dilemma_situation = new_row['DILEMMA']
    instruction_prompt = create_prompt_template_for_reasoning_eval_natural_behavior()
    prompt = f'{instruction_prompt}{dilemma_situation}'

    if args.reasoning:
        resp, input_tokens, output_tokens, reasoning_tokens, cot_thinking_trace = collect_thinking_response(
            client, args.model, prompt, args.api_provider, args.budget_tokens, args.reasoning_effort
        )
    else:
        resp, input_tokens, output_tokens = collect_response(client, args.model, prompt, args.api_provider)
        cot_thinking_trace = ""
        reasoning_tokens = -1
    
    new_row["RUBRIC"] = literal_eval(new_row["RUBRIC"])
    new_row['thinking_trace'] = cot_thinking_trace
    new_row['model_resp'] = resp
    new_row['input_tokens'] = input_tokens
    new_row['output_tokens'] = output_tokens
    new_row['reasoning_tokens'] = reasoning_tokens
    new_row['idx'] = idx
    new_row['model'] = args.model
    
    write_to_jsonl(new_row, output_file_jsonl)
    return new_row


# Load data
ds = load_dataset("morebench/morebench", token=args.hf_token, data_files="morebench_public.csv", split="train")
df = ds.to_pandas()

df = df[df['THEORY'] == 'neutral']

if args.debug:
    df = df[:5]

# Skip already processed rows
existing_idx = load_existing_indices(output_file_jsonl)
if existing_idx:
    print(f"Found {len(existing_idx)} existing rows out of {len(df)}")
    df = df[~df.index.isin(existing_idx)]

# Process rows
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_parallel_request) as executor:
    futures = [executor.submit(process_single_row, row, idx) for idx, row in df.iterrows()]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            future.result()
        except Exception as e:
            print(f"Error: {e}")