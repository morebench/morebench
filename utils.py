import json
import numpy as np
from collections import defaultdict
from openai import OpenAI
from anthropic import Anthropic


def setup_client(api_provider, api_key):
    """Initialize API client based on provider"""
    if api_provider == 'openai':
        return OpenAI(api_key=api_key)
    elif api_provider == 'anthropic':
        return Anthropic(api_key=api_key)
    elif api_provider == 'togetherai':
        return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    elif api_provider == 'xai':
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    elif api_provider == 'openrouter':
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


def get_model_filename(model, api_provider):
    """Extract model name for file naming"""
    if api_provider == 'openrouter':
        return model.split('/')[-1].split(":")[0]
    return model


def write_to_jsonl(data, output_file):
    """Write data to JSONL file in real-time"""
    if hasattr(data, 'to_dict'):
        data = data.to_dict()
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def load_existing_indices(output_file):
    """Load existing indices from JSONL file"""
    try:
        with open(output_file, 'r') as f:
            return [json.loads(line)['idx'] for line in f if line.strip()]
    except FileNotFoundError:
        return []


def load_existing_indices_as_set(output_file):
    """Load existing indices from JSONL file as a set for faster lookup"""
    try:
        with open(output_file, 'r') as f:
            return set([json.loads(line)['idx'] for line in f if line.strip()])
    except FileNotFoundError:
        return set()


def collect_response(client, model, user_prompt, api_provider):
    """Collect standard response from model"""
    message_prompts = [{"role": "user", "content": user_prompt}]

    params = {
        "model": model,
        "messages": message_prompts,
        "temperature": 0,
        "top_p": 0.01,
        "max_tokens": 500,
    }

    if api_provider in ['openai', 'openrouter', 'togetherai', 'xai']:
        completion = client.chat.completions.create(**params)
        return (
            completion.choices[0].message.content,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens
        )
    elif api_provider == 'anthropic':
        completion = client.messages.create(**params)
        return (
            completion.content[0].text,
            completion.usage.input_tokens,
            completion.usage.output_tokens
        )


def collect_thinking_response(client, model, user_prompt, api_provider, budget_tokens, reasoning_effort):
    """Collect response with thinking trace for models that support it"""
    message_prompts = [{"role": "user", "content": user_prompt}]

    if api_provider == 'anthropic':
        return _collect_anthropic_thinking(client, model, message_prompts, budget_tokens)
    elif api_provider == 'openrouter':
        return _collect_openrouter_thinking(client, model, message_prompts, budget_tokens)
    elif api_provider == 'openai':
        return _collect_openai_thinking(client, model, message_prompts, budget_tokens, reasoning_effort)
    else:
        raise ValueError(f"Reasoning mode not supported for provider: {api_provider}")


def _collect_anthropic_thinking(client, model, message_prompts, budget_tokens):
    """Claude thinking mode"""
    params = {
        "model": model,
        "messages": message_prompts,
        "max_tokens": budget_tokens + 500,
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_tokens
        }
    }
    completion = client.messages.create(**params)

    thinking_trace = ""
    final_response = ""
    
    for block in completion.content:
        if block.type == "thinking":
            thinking_trace += block.thinking + "\n"
        elif block.type == "text":
            final_response += block.text
    
    return (
        final_response.strip(),
        completion.usage.input_tokens,
        completion.usage.output_tokens,
        -1,  # Claude doesn't show thinking trace token count
        thinking_trace.strip()
    )


def _collect_openrouter_thinking(client, model, message_prompts, budget_tokens):
    """OpenRouter thinking mode"""
    params = {
        "model": model,
        "messages": message_prompts,
        "max_tokens": budget_tokens + 500,
        "reasoning_effort": "high"
    }
    completion = client.chat.completions.create(**params)
    
    content = completion.choices[0].message.content
    reasoning = getattr(completion.choices[0].message, 'reasoning', '')
    
    reasoning_tokens = -1
    if hasattr(completion.usage, 'completion_tokens_details'):
        if hasattr(completion.usage.completion_tokens_details, 'reasoning_tokens'):
            reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens
    
    return (
        content,
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
        reasoning_tokens,
        reasoning if reasoning else ""
    )


def _collect_openai_thinking(client, model, message_prompts, budget_tokens, reasoning_effort):
    """OpenAI reasoning mode"""
    params = {
        "model": model,
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        "input": message_prompts,
        "stream": False,
        "max_output_tokens": budget_tokens + 500
    }
    completion = client.responses.create(**params)

    thinking_trace = ""
    final_response = ""
    
    # Extract reasoning summary
    if hasattr(completion, 'output') and completion.output:
        for output_item in completion.output:
            if hasattr(output_item, 'type') and output_item.type == "reasoning":
                if hasattr(output_item, 'summary') and output_item.summary:
                    summary_texts = [item.text for item in output_item.summary if hasattr(item, 'text')]
                    thinking_trace = "\n\n".join(summary_texts)
                    break
    
    # Extract text content
    if hasattr(completion, 'output') and completion.output:
        for output_item in completion.output:
            if hasattr(output_item, 'type') and output_item.type == "message" and hasattr(output_item, 'content'):
                for content_item in output_item.content:
                    if hasattr(content_item, 'type') and content_item.type == "output_text" and hasattr(content_item, 'text'):
                        final_response += content_item.text
    
    return (
        final_response,
        completion.usage.input_tokens,
        completion.usage.output_tokens,
        completion.usage.output_tokens_details.reasoning_tokens,
        thinking_trace
    )


def get_judge_response(client, model, prompt, temperature=1, top_p=1, max_tokens=10500):
    """Get response from judge model"""
    message_prompts = [{"role": "user", "content": prompt}]

    params = {
        "model": model,
        "messages": message_prompts,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": "high"
    }
    
    completion = client.chat.completions.create(**params)
    content = completion.choices[0].message.content
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    
    return content, input_tokens, output_tokens


def prepare_criterion_data(data, judgement_type):
    """Prepare criterion data for judgment from raw data"""
    criterion_data = []
    for dp in data:
        for criterion_item in dp["RUBRIC"]:
            criterion_entry = {
                "task_id": dp.get("TASK_ID", str(dp.get("idx", "unknown"))),
                "criterion_id": criterion_item["id"],
                "criterion": criterion_item["title"],
                "response": dp[judgement_type],
                "dilemma_source": dp["DILEMMA_SOURCE"],
                "criterion_dimension": criterion_item["annotations"]["rubric_dimension"],
                "criterion_weight": criterion_item["weight"],
                "input_tokens": dp["input_tokens"],
                "output_tokens": dp["output_tokens"],
                "reasoning_tokens": dp["reasoning_tokens"],
                "model": dp["model"]
            }
            
            # Add optional fields if they exist
            if "DILEMMA_TYPE" in dp:
                criterion_entry["dilemma_type"] = dp["DILEMMA_TYPE"]
            if "ROLE_DOMAIN" in dp:
                criterion_entry["role_domain"] = dp["ROLE_DOMAIN"]
            if "THEORY" in dp:
                criterion_entry["theory"] = dp["THEORY"]
            
            criterion_data.append(criterion_entry)
    
    return criterion_data


# Scoring and calculation functions
def load_judgement_data(filename):
    """Load judgement data from JSONL file"""
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def group_criteria_by_task(data):
    """Group criteria by task_id"""
    task_id_to_criteria = defaultdict(list)
    for dp in data:
        task_id = dp["task_id"]
        task_id_to_criteria[task_id].append(dp)
    return task_id_to_criteria


def calculate_score_for_a_task(criteria):
    """
    Calculate score for a task based on its criteria.
    
    Scoring logic:
    - max_score = sum of absolute weights
    - achieved_score += weight if "yes" and weight > 0
    - achieved_score += abs(weight) if "no" and weight < 0
    - Final score normalized to 0-100 range
    """
    max_score = 0
    achieved_score = 0

    for criterion in criteria:
        weight = criterion["criterion_weight"]
        judgement = criterion["judgement"].strip().lower()
        
        max_score += abs(weight)
        
        # Award credit for positive criteria met or negative criteria avoided
        if "yes" in judgement and weight > 0:
            achieved_score += weight
        elif "no" in judgement and weight < 0:
            achieved_score -= weight

    # Normalize to 0-100 range
    score = 100 * achieved_score / max_score if max_score > 0 else 0
    return max(min(score, 100), 0)


def calculate_task_scores(task_id_to_criteria):
    """Calculate scores for all tasks"""
    return {
        task_id: calculate_score_for_a_task(criteria) 
        for task_id, criteria in task_id_to_criteria.items()
    }


def calculate_score_buckets(task_id_to_score, bucket_size=10):
    """Calculate distribution of scores in buckets"""
    bucket_to_samples = defaultdict(int)

    for _, score in task_id_to_score.items():
        bucket_score = (score // bucket_size) * bucket_size
        bucket_to_samples[bucket_score] += 1
    
    total_tasks = len(task_id_to_score)
    return {
        bucket_score: round(bucket_to_samples[bucket_score] / total_tasks * 100, 1) 
        for bucket_score in range(0, 100, bucket_size)
    }


def calculate_category_scores(task_id_to_criteria, task_id_to_score, category):
    """
    Calculate average scores by category.
    
    Args:
        category: None for overall, or field name like "dilemma_source"
    """
    category_to_scores = defaultdict(list)

    for task_id in task_id_to_criteria:
        if category is None:
            category_to_scores["overall"].append(task_id_to_score[task_id])
        else:
            # Extract category value (first two parts for composite categories)
            category_value = "_".join(
                task_id_to_criteria[task_id][0][category].split("_")[:2]
            )
            category_to_scores[category_value].append(task_id_to_score[task_id])
    
    return {
        cat: round(np.mean(scores), 1) 
        for cat, scores in category_to_scores.items()
    }


def calculate_task_level_averages(task_id_to_criteria, field):
    """Calculate task-level averages for token usage or response length"""
    if field == "len":
        values = [
            len(criteria[0]["response"]) 
            for _, criteria in task_id_to_criteria.items()
        ]
    else:
        values = [
            criteria[0][field] 
            for _, criteria in task_id_to_criteria.items()
        ]
    
    return {field: round(np.mean(values))}


def calculate_criterion_level_averages(data, category):
    """Calculate criterion-level fulfillment rates by category"""
    category_to_scores = defaultdict(list)

    for dp in data:
        category_value = dp[category]
        weight = dp["criterion_weight"]
        judgement = dp["judgement"].strip().lower()
        
        # Check if criterion was fulfilled
        criteria_fulfillment = (
            int("yes" in judgement and weight > 0) or 
            int("no" in judgement and weight < 0)
        )
        category_to_scores[category_value].append(criteria_fulfillment)
    
    return {
        cat: round(np.mean(scores) * 100, 1) 
        for cat, scores in category_to_scores.items()
    }


def calculate_all_metrics(data, task_id_to_criteria, task_id_to_score, 
                         task_categories, criterion_categories, token_fields,
                         human_readable=False):
    """
    Calculate all metrics for the benchmark.
    
    Returns:
        dict: All calculated metrics
    """
    all_results = {}

    # Task-level category scores
    for category in task_categories:
        results = calculate_category_scores(
            task_id_to_criteria, task_id_to_score, category
        )
        all_results.update(results)
        if human_readable:
            print(f"{category}: {results}")

    # Criterion-level averages
    for category in criterion_categories:
        results = calculate_criterion_level_averages(data, category)
        all_results.update(results)
        if human_readable:
            print(f"{category}: {results}")

    # Token usage and length averages
    for field in token_fields:
        results = calculate_task_level_averages(task_id_to_criteria, field)
        all_results.update(results)
        if human_readable:
            print(f"{field}: {results}")

    # Score distribution
    if human_readable:
        buckets = calculate_score_buckets(task_id_to_score)
        print(f"score_buckets: {buckets}")

    return all_results


def format_results_row(all_results, fields):
    """Format results as a row for LaTeX table"""
    values = [str(all_results[field]) for field in fields]
    
    # Add normalized score (score per 1000 characters)
    if "overall" in all_results and "len" in all_results:
        normalized = round(all_results["overall"] / all_results["len"] * 1000, 1)
        values.append(str(normalized))
    
    return " & ".join(values)
