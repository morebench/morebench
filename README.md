# MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes

**Evaluating how well language models reason through complex moral dilemmas.**

MoReBench tests LLMs on 1000 real-world ethical scenarios across diverse domains (healthcare, business, technology, law) and ethical frameworks (consequentialism, deontology, virtue ethics). Unlike current tasks focusing on the outcome accuracy (e.g. math and code), MoReBench evaluates the *quality of moral reasoning* through multi-dimensional rubrics.

<p align="center">
  <a href="https://morebench.github.io/">Project Website</a> | 
  <a href="https://arxiv.org/abs/2510.16380">Paper</a> | 
  <a href="https://huggingface.co/datasets/morebench/morebench">Dataset</a> | 
  <a href="link">Leaderboard - Coming soon</a> | 
  <a href="link">UK AISI Inspect Eval - Coming soon</a>
</p>


![pipeline](img/intro_pipeline_img.png)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/morebench/morebench.git
cd morebench

# create a new virtual environment
conda create -n morebench python=3.12 -y
conda activate morebench

# Install dependencies
pip install -r requirements.txt

# 1. Generate model responses
python run_inferences_on_dilemmas.py \
    -ap openai \
    -ak YOUR_API_KEY \
    -m gpt-4o \
    -ht YOUR_HUGGINGFACE_TOKEN

# 2. Evaluate with judge model
python run_best_judge_on_responses.py \
    -i generations/gpt-4o_reasoning_medium_seed_0.jsonl \
    -ak YOUR_OPENROUTER_KEY \
    -jt model_resp
    
# 3. Calculate benchmark scores
python calculate_morebench.py \
    -i model_resp_judgements/gpt-4o_reasoning_medium_seed_0.jsonl \
    -f human
```

**Output**: Scores across 5 dimensions: identifying (moral factors), logical process, clear process, helpful outcome, harmless outcome

---

## How It Works

```
┌─────────────────┐
│  Moral Dilemma  │  "As a oversight Scientist AI,... Should I report it or not?"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Response │  Extended reasoning + final decision
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Judge Scoring  │  Evaluate against 20-47 rubric criteria
└────────┬────────┘
         │
         ▼
┌─────────────────┐  
│    Calculate    |  Aggregate by dimensions + overall 
|      Score      |  1. MoReBench-Regular: Raw score
|  (Regular/Hard) │  2. MoReBench-Hard: Length-controlled score
└─────────────────┘  
```

### Two Dataset Variants

| Dataset | Size | Description |
|---------|------|-------------|
| **MoReBench** | 1000 | Procedural reasoning on diverse moral scenarios |
| **MoReBench-Theory** | 150 | Procedural reasoning under five moral frameworks (Kantian Deontology, Benthamite Act Utilitarianism, Aristotelian Virtue Ethics, Scanlonian Contractualism, and Gauthierian Contractarianism) |

---

## Complete Pipeline

### 1. Generate Responses

#### MoReBench Dataset (1000 samples)

```bash
python run_inferences_on_dilemmas.py \
    -ap openai \
    -ak $OPENAI_KEY \
    -m o4-mini \
    -r \
    -re high \
    -ht YOUR_HUGGINGFACE_TOKEN
```

#### MoReBench-Theory Dataset (150 samples)

```bash
python run_inferences_on_dilemmas_theory.py \
    -ap openai \
    -ak $OPENAI_KEY \
    -m o4-mini \
    -r \
    -re high \
    -ht YOUR_HUGGINGFACE_TOKEN
```

### 2. Judge Responses

You can judge both the final response (`model_resp`) and the thinking process (`thinking_trace`):

#### Judge Final Responses

```bash
# MoReBench dataset
python run_best_judge_on_responses.py \
    -i generations/gpt-4o_reasoning_high_seed_0.jsonl \
    -ak $OPENROUTER_KEY \
    -jt model_resp

# MoReBench-Theory dataset
python run_best_judge_on_responses_theory.py \
    -i generations_theory/gpt-4o_reasoning_high.jsonl \
    -ak $OPENROUTER_KEY \
    -jt model_resp
```

#### Judge Thinking Traces

```bash
# MoReBench dataset
python run_best_judge_on_responses.py \
    -i generations/gpt-4o_reasoning_high_seed_0.jsonl \
    -ak $OPENROUTER_KEY \
    -jt thinking_trace

# MoReBench-Theory dataset
python run_best_judge_on_responses_theory.py \
    -i generations_theory/gpt-4o_reasoning_high.jsonl \
    -ak $OPENROUTER_KEY \
    -jt thinking_trace
```

### 3. Calculate Benchmark Scores

#### MoReBench Dataset Results

```bash
# Human-readable output
python calculate_morebench.py \
    -i model_resp_judgements/gpt-4o_reasoning_high_seed_0.jsonl \
    -f human

# LaTeX table row (for papers)
python calculate_morebench.py \
    -i model_resp_judgements/gpt-4o_reasoning_high_seed_0.jsonl \
    -f latex
```

**Human-readable output:**
```
None: {'overall': 75.1}
dilemma_source: {'daily_dilemmas': 75.2, 'ai_risk': 68.9, 'expert_case': 82.1}
role_domain: {'ai_advisor': 73.4, 'ai_agent': 76.8}
dilemma_type: {'short_case': 71.5, 'long_case': 79.3}
criterion_dimension: {'identifying': 78.2, 'logical process': 72.5, 'clear process': 70, 'helpful outcome': 70, 'harmless outcome': 70 ...}
criterion_weight: {1: 74.3, 2: 76.8, 3: 75.1}
input_tokens: {'input_tokens': 850}
output_tokens: {'output_tokens': 450}
len: {'len': 1250}

=== Summary ===
Overall Score: 75.1
Average Response Length: 1250 chars
Normalized Score (per 1k chars): 60.1
```

**LaTeX output:**
```
75.2 & 68.9 & 82.1 & 71.5 & 79.3 & 73.4 & 76.8 & 75.1 & 1250 & 60.1
```

#### Theory Dataset Results

```bash
# Human-readable output
python calculate_morebench_theory.py \
    -i model_resp_judgements_theory/gpt-4o_reasoning_high.jsonl \
    -f human

# LaTeX table row
python calculate_morebench_theory.py \
    -i model_resp_judgements_theory/gpt-4o_reasoning_high.jsonl \
    -f latex
```

---

## Script Reference

### Generation Scripts

#### `run_inferences_on_dilemmas.py`

Generate responses on main dataset (500 neutral dilemmas).

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--api_provider` | `-ap` | ✅ | - | openai, anthropic, openrouter, togetherai, xai |
| `--api_key` | `-ak` | ✅ | - | API key for the provider |
| `--model` | `-m` | ✅ | - | Model identifier |
| `--reasoning` | `-r` | ❌ | False | Enable thinking/reasoning mode |
| `--budget_tokens` | `-b` | ❌ | 10000 | Token budget for reasoning |
| `--reasoning_effort` | `-re` | ❌ | medium | minimal/low/medium/high |
| `--seed` | `-s` | ❌ | 0 | Random seed |
| `--num_parallel_request` | `-n` | ❌ | 100 | Concurrent requests |
| `--input_file` | `-i` | ❌ | dataset_11092025.csv | Input CSV |
| `--generations_dir` | `-g` | ❌ | generations | Output directory |
| `--debug` | `-d` | ❌ | False | Test with 5 samples |

#### `run_inferences_on_dilemmas_theory.py`

Generate responses on theory dataset (150 framework-specific dilemmas).

Same arguments as above, except:
- Default `input_file`: dataset_16092025.csv
- Default `generations_dir`: generations_theory
- No `--seed` argument

### Judgment Scripts

#### `run_best_judge_on_responses.py`

Evaluate responses using judge model (expects 500 samples × ~23 criteria = 11,568 judgements).

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input_file` | `-i` | ✅ | - | Generated responses JSONL |
| `--api_key` | `-ak` | ✅ | - | OpenRouter API key |
| `--judgement_type` | `-jt` | ❌ | thinking_trace | model_resp or thinking_trace |
| `--judge_model` | `-jm` | ❌ | openai/gpt-oss-120b | Judge model |
| `--num_parallel_request` | `-n` | ❌ | 100 | Concurrent requests |
| `--expected_samples` | `-es` | ❌ | 500 | Expected sample count |
| `--output_dir` | `-o` | ❌ | auto | Custom output location |
| `--debug` | `-d` | ❌ | False | Test with 10 samples |

#### `run_best_judge_on_responses_theory.py`

Evaluate theory dataset responses (expects 150 samples × ~25 criteria = 3,835 judgements).

Same arguments as above, except default `num_parallel_request`: 160 and `expected_samples`: 150.

### Calculation Scripts

#### `calculate_morebench.py`

Calculate benchmark scores from judgment data (main dataset).

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input_file` | `-i` | ✅ | - | Judgement JSONL file |
| `--format` | `-f` | ❌ | latex | latex or human |
| `--expected_samples` | `-es` | ❌ | 11568 | Expected judgement count |

**Scoring Logic:**
- Each task has 6-10 weighted criteria
- `max_score = sum(|weight|)` for all criteria
- `achieved_score += weight` if criterion met (yes/no based on weight sign)
- `task_score = 100 × achieved_score / max_score` (clamped to 0-100)
- Final scores aggregated by category (dilemma source, role domain, etc.)

**Output Fields (LaTeX format):**
`daily_dilemmas & ai_risk & expert_case & short_case & long_case & ai_advisor & ai_agent & overall & length & normalized`

#### `calculate_morebench_theory.py`

Calculate benchmark scores from judgment data (theory dataset).

Same arguments as `calculate_morebench.py`, except default `expected_samples`: 3835.

**Output Fields (LaTeX format):**
`Gauthierian_Contractarianism & Scanlonian_Contractualism & Act_Utilitarianism & Aristotelian_Virtue_Ethics & Kantian_Deontology & overall & length & normalized`

---

## Output Files

### Generation Phase

**Format**: `{model}_reasoning_{effort}_seed_{seed}.jsonl`

```jsonl
{
  "TASK_ID": "task_001",
  "DILEMMA": "A doctor must choose...",
  "RUBRIC": [...],
  "model_resp": "After careful consideration...",
  "thinking_trace": "Let me analyze this step by step...",
  "input_tokens": 850,
  "output_tokens": 450,
  "reasoning_tokens": 3200,
  "model": "gpt-4o",
  "idx": 0
}
```

### Judgment Phase

**Format**: `{judgement_type}_judgements/{filename}` (`{judgement_type}_judgements_theory` for theory dataset)

```jsonl
{
  "task_id": "task_001",
  "criterion_id": "crit_1",
  "criterion": "Thoroughness of analysis",
  "response": "After careful consideration...",
  "judgement": "yes",
  "judge_input_tokens": 920,
  "judge_output_tokens": 180,
  "criterion_weight": 2,
  "criterion_dimension": "thoroughness",
  ...
}
```

### Calculation Phase

**Human format**: Detailed breakdown by category + summary
**LaTeX format**: Single row for tables (values separated by `&`)

---

## Understanding the Scores

### Overall Score (0-100)
The primary metric representing quality of moral reasoning across all criteria.


### MoReBench-Easy (Raw Score)
Score per 1,000 characters.

### MoReBench-Hard (Normalized Score)
Score per 1,000 characters (controls for response length bias).

---

## Troubleshooting

**Issue**: API rate limits  
**Solution**: Reduce `-n` parameter: `-n 10`

**Issue**: Inference script raises errors and cannot complete the whole file
**Solution**: Rerun the script again. The script will start from what missing.


---

## Citation

```bibtex
@misc{chiu2025morebenchevaluatingproceduralpluralistic,
        title={MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes}, 
        author={Yu Ying Chiu and Michael S. Lee and Rachel Calcott and Brandon Handoko and Paul de Font-Reaulx and Paula Rodriguez and Chen Bo Calvin Zhang and Ziwen Han and Udari Madhushani Sehwag and Yash Maurya and Christina Q Knight and Harry R. Lloyd and Florence Bacus and Mantas Mazeika and Bing Liu and Yejin Choi and Mitchell L Gordon and Sydney Levine},
        year={2025},
        eprint={2510.16380},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2510.16380}, 
  }
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---


**Questions?** Open an issue or contact [Yu Ying Chiu (Kelly Chiu)](mailto:kellycyy@uw.edu)
