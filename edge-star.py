# -*- coding: utf-8 -*-
import time
import os
import re
import json
import openai
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from itertools import cycle

MAIN_LLM_URLS = [f"http://localhost:{port}/v1" for port in [8000, 8001, 8002, 8003]]
TOOL_LLM_URLS = MAIN_LLM_URLS
LOCAL_MODEL_NAME = "qwen3-4b-fp8"  

MAX_WORKERS = 12                  
MAX_PATHS_TO_PROCESS = 5000       

INPUT_AS_PATHS_FILE = "/home/u2021110842/autodl-tmp/BGPAgent/asrank_data/20240301_all-paths_cache_top_5000.json"
AS_INFO_DB_FILE = "/home/u2021110842/autodl-tmp/BGPAgent/get_asrank_api_result/asn_infor_TOP5000.json"
GROUND_TRUTH_DB_FILE = "/home/u2021110842/autodl-tmp/BGPAgent/get_asrank_api_result/ground_truth2.json"

date_str = datetime.today().strftime('%Y%m%d')
output_dir = f"finetune_data_{date_str}_golden_samples"
os.makedirs(output_dir, exist_ok=True)

FINETUNE_GOLDEN_SAMPLES_FILE = os.path.join(output_dir, "finetune_samples.jsonl")
LOG_FILE = os.path.join(output_dir, "processing_log.txt")
ENABLE_DEBUG_OUTPUTS = True
if ENABLE_DEBUG_OUTPUTS:
    DEBUG_DIR = f"debug_logs_{date_str}_golden_samples"
    os.makedirs(DEBUG_DIR, exist_ok=True)


BGP_GOLDEN_PROMPT_TEMPLATE = """You are a world-class expert in BGP (Border Gateway Protocol). Your mission is to generate a perfect, detailed analysis for a given AS Path, following the provided example and instructions.

**CRITICAL INSTRUCTIONS:**
1.  **REASONING (Chain-of-Thought):** Provide a detailed, step-by-step reasoning that logically explains and justifies the provided "Expert Findings". Your reasoning must be based on the "Additional AS Information" and general BGP principles like valley-free routing.
2.  **FINAL JSON OUTPUT (MANDATORY):** After your complete reasoning, you MUST conclude your entire response with the final answer as a single, valid JSON list of strings. This list **MUST EXACTLY** match the "Expert Findings" provided in the task section.

--- EXAMPLE START ---

**Input AS Path:**
15562|2914|58453|9808

**Additional AS Information:**
AS 15562: {"as_information": {"orgName": "SNIJDERS"}}
AS 2914: {"as_information": {"orgName": "NTT Communications Corporation"}}
AS 58453: {"as_information": {"orgName": "China Mobile International Limited"}}
AS 9808: {"as_information": {"orgName": "China Mobile Guangdong"}}

**Example Expert Analysis Output:**
Here is my analysis of the AS path 15562|2914|58453|9808.

1.  **15562-2914:** AS15562 (SNIJDERS) is a smaller network, while AS2914 (NTT) is a well-known Tier-1 global provider. Based on their relative sizes and roles in the internet ecosystem, it is highly probable that AS15562 is a customer of AS2914, purchasing transit services to reach the global internet. This is a classic Customer-to-Provider (C2P) relationship.

2.  **2914-58453:** AS2914 (NTT) and AS58453 (China Mobile International) are both massive, global-scale carriers. They operate at a similar, top-tier level in the internet hierarchy. Their relationship is most likely Peer-to-Peer (P2P), where they agree to exchange traffic between their respective customers for mutual benefit, typically without settlement fees.

3.  **58453-9808:** AS58453 is the international backbone for China Mobile, while AS9808 is a regional branch (Guangdong). The international entity acts as a provider of global connectivity to its own regional branches. Therefore, AS58453 is the provider and AS9808 is the customer. This is a Provider-to-Customer (P2C) relationship.

The overall path adheres to the valley-free routing policy: an upward step from a customer to a provider (C2P), a traversal across a peer link (P2P), and a downward step to a regional customer (P2C).

["15562-2914: C2P", "2914-58453: P2P", "58453-9808: P2C"]
--- EXAMPLE END ---

--- YOUR TASK ---

**CONTEXT:** An expert panel has already determined the definitive business relationships for the AS Path below. Your task is to generate the reasoning for their findings. **DO NOT** mention "Expert Findings", "Ground Truth", "hints", or this instruction in your reasoning.

**Input AS Path:**
<AS_PATH_PLACEHOLDER>

**Additional AS Information:**
<AS_INFO_PLACEHOLDER>

**Expert Findings (This is the Ground Truth you must justify and output):**
<GROUND_TRUTH_PLACEHOLDER>

---
Now, provide your expert analysis for the new AS path, following all instructions and the example format precisely.
"""


log_lock = threading.Lock()
save_lock = threading.Lock()

def load_processed_paths(*jsonl_files):
    seen = set()
    pattern = re.compile(r"Input AS Path[:*]*\s*([0-9]+(?:\|[0-9]+)*)")
    for fp in jsonl_files:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    instr = entry.get("instruction", "")
                    m = pattern.search(instr)
                    if m:
                        seen.add(m.group(1))
                except json.JSONDecodeError:
                    continue
    return seen

def log_message(message):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

def load_json_file(filename):
    if not os.path.exists(filename):
        log_message(f"严重错误: 文件 {filename} 不存在。程序将退出。")
        exit(1)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_message(f"严重错误: JSON文件 {filename} 格式错误: {e}")
        exit(1)

def save_to_finetune_dataset(file_path, prompt, completion):
    with save_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            finetune_entry = {"instruction": prompt, "input": "", "output": completion}
            f.write(json.dumps(finetune_entry, ensure_ascii=False) + "\n")

def save_debug_log(as_path, debug_data):
    if not ENABLE_DEBUG_OUTPUTS: return
    fname = as_path.replace('|', '_').replace('/', '_') + '.json'
    with open(os.path.join(DEBUG_DIR, fname), 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)

def get_llm_response(client, model_name, prompt, temperature=0.1):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature, max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"LLM call failed at {client.base_url} with model {model_name}: {e}")
        return None

def extract_json_list_from_text(text: str):
    if not text:
        return None, ""
    start_pos = text.rfind('[')
    if start_pos == -1:
        return None, text
    snippet = text[start_pos:]
    balance = 0
    end_index_in_snippet = -1
    for i, char in enumerate(snippet):
        if char == '[': balance += 1
        elif char == ']': balance -= 1
        if balance == 0:
            end_index_in_snippet = i
            break
    if end_index_in_snippet != -1:
        end_pos_in_text = start_pos + end_index_in_snippet + 1
        json_str = text[start_pos:end_pos_in_text]
        cot = text[:start_pos].strip()
        try:
            parsed_list = json.loads(json_str)
            if isinstance(parsed_list, list):
                return parsed_list, cot
        except json.JSONDecodeError:
            pass 
    return None, text


def verify_relationships(inferred, ground_truth):
    if inferred is None:
        return ["Parsing error: No relationship list was extracted from the model's output."]
    if set(inferred) != set(ground_truth):
        return [f"Mismatch Error: Output '{inferred}' does not match ground truth '{ground_truth}'."]
    return []

def extract_facts_from_cot(cot, client_tool):
    if not cot: return []
    prompt = f"""Your task is to extract all mentions of Autonomous System Numbers (ASNs) and their corresponding organization names from the provided text.
**Input Text:**\n---\n{cot}\n---\n
**Output Requirements:**
- You MUST return the result as a single, valid JSON list of lists. Example: `[[1239, "Sprint"], [3356, "Lumen"]]`.
- If no pairs are found, you MUST return an empty list `[]`.
- Your entire response MUST ONLY be the JSON list.
**JSON Output:**"""
    resp = get_llm_response(client_tool, LOCAL_MODEL_NAME, prompt, temperature=0.0)
    facts, _ = extract_json_list_from_text(resp)
    if facts and isinstance(facts, list):
        return [[str(item[0]), item[1]] for item in facts if isinstance(item, list) and len(item) == 2]
    return []

def verify_facts_in_batch_llm(extracted_facts, as_info_db, client_tool):
    errors = []
    verification_pairs = []
    for asn, org_from_cot in extracted_facts:
        true_org_name = as_info_db.get(asn, {}).get('as_information', {}).get('orgName')
        if not true_org_name or org_from_cot.strip().lower() == true_org_name.strip().lower():
            continue
        verification_pairs.append({"asn": asn, "name_from_cot": org_from_cot, "name_from_db": true_org_name})
    if not verification_pairs: return []

    pairs_json_str = json.dumps(verification_pairs, indent=2, ensure_ascii=False)
    prompt = f"""You are a fact-checking assistant. For each JSON object below, determine if `name_from_cot` and `name_from_db` refer to the same entity.
**Input:**\n{pairs_json_str}
**Task:**
Respond with a single JSON list of booleans (`true` for same, `false` for different), corresponding to the input list.
**Mandatory Output Format:**
Your entire response must be ONLY the JSON list. Example: `[true, false, true]`
**JSON Output:**"""
    resp = get_llm_response(client_tool, LOCAL_MODEL_NAME, prompt, temperature=0.0)
    results, _ = extract_json_list_from_text(resp)
    if results and isinstance(results, list) and len(results) == len(verification_pairs):
        for i, is_match in enumerate(results):
            if not is_match:
                pair = verification_pairs[i]
                errors.append(f"Factual error for AS{pair['asn']}: model reasoned with '{pair['name_from_cot']}', but DB shows '{pair['name_from_db']}'.")
    else:
        errors.append("Fact verification call failed or returned malformed data.")
    return errors


def build_perfect_prompt(as_path, as_info_db, ground_truth_list):
    asn_list = [x.strip() for x in as_path.split("|") if x.strip()]
    info_lines = []
    for asn in asn_list:
        org_name = as_info_db.get(asn, {}).get('as_information', {}).get('orgName', 'No data')
        info_lines.append(f'AS {asn}: {{"as_information": {{"orgName": "{org_name}"}}}}')
    as_info_text = "\n".join(info_lines)
    ground_truth_text = "\n".join([f"- {rel}" for rel in ground_truth_list])
    prompt = BGP_GOLDEN_PROMPT_TEMPLATE.replace("<AS_PATH_PLACEHOLDER>", as_path)
    prompt = prompt.replace("<AS_INFO_PLACEHOLDER>", as_info_text)
    prompt = prompt.replace("<GROUND_TRUTH_PLACEHOLDER>", ground_truth_text)
    return prompt

def process_path(as_path, as_info_db, ground_truth_db, client_main, client_tool):
    debug_log = {'as_path': as_path}

    nums = [x.strip() for x in as_path.split("|") if x.strip()]
    for asn in nums:
        if asn not in as_info_db:
            log_message(f"SKIPPING: AS '{asn}' in path '{as_path}' not in info DB.")
            return

    ground_truth = []
    for i in range(len(nums) - 1):
        key = f"{nums[i]}-{nums[i+1]}"
        rel = ground_truth_db.get(key)
        if not rel or rel.lower() == 'none':
            log_message(f"SKIPPING: Ground truth for '{key}' in path '{as_path}' is missing.")
            return
        ground_truth.append(f"{key}: {rel.upper()}")
    debug_log['ground_truth'] = ground_truth

    prompt = build_perfect_prompt(as_path, as_info_db, ground_truth)
    debug_log['prompt'] = prompt
    log_message(f"Processing path '{as_path}'...")
    raw_output = get_llm_response(client_main, LOCAL_MODEL_NAME, prompt)
    if not raw_output:
        log_message(f"FAILURE: LLM call failed for path '{as_path}'.")
        debug_log['outcome'] = 'Failed (LLM call failed)'
        save_debug_log(as_path, debug_log)
        return

    debug_log['raw_output'] = raw_output

    inferred_relations, cot = extract_json_list_from_text(raw_output)
    relation_errors = verify_relationships(inferred_relations, ground_truth)
    extracted_facts = extract_facts_from_cot(cot, client_tool)
    fact_errors = verify_facts_in_batch_llm(extracted_facts, as_info_db, client_tool)
    all_errors = relation_errors + fact_errors
    debug_log['errors'] = all_errors

    if not all_errors:
        log_message(f"SUCCESS: Path '{as_path}' passed all checks. Saving to dataset.")
        save_to_finetune_dataset(FINETUNE_GOLDEN_SAMPLES_FILE, prompt, raw_output)
        debug_log['outcome'] = 'Success'
    else:
        log_message(f"FAILURE: Path '{as_path}' failed verification. Errors: {all_errors}. Discarding.")
        debug_log['outcome'] = 'Failed (Verification failed)'
    
    save_debug_log(as_path, debug_log)


def main():
    log_message("===== Star Sample Generation Script Started =====")

    main_clients = [openai.OpenAI(api_key="none", base_url=url, timeout=300.0) for url in MAIN_LLM_URLS]
    tool_clients = [openai.OpenAI(api_key="none", base_url=url, timeout=300.0) for url in TOOL_LLM_URLS]
    log_message(f"Initialized {len(main_clients)} LLM clients for URLs: {MAIN_LLM_URLS}")
    log_message("Loading databases...")
    as_info_list = load_json_file(AS_INFO_DB_FILE)
    as_info_db = {item['asn']: item for item in as_info_list} if isinstance(as_info_list, list) else as_info_list
    ground_truth_db = load_json_file(GROUND_TRUTH_DB_FILE)
    log_message(f"Loading AS paths from '{INPUT_AS_PATHS_FILE}'...")
    with open(INPUT_AS_PATHS_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            input_paths = [d['as_path'] for d in data if isinstance(d, dict) and 'as_path' in d]
        except json.JSONDecodeError:
            f.seek(0)
            input_paths = []
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and 'as_path' in obj:
                        input_paths.append(obj['as_path'])
                except (json.JSONDecodeError, AttributeError):
                    continue
    
    if not input_paths:
        log_message(f"CRITICAL ERROR: No valid AS paths could be loaded from '{INPUT_AS_PATHS_FILE}'. Exiting.")
        return
    processed_paths = load_processed_paths(FINETUNE_GOLDEN_SAMPLES_FILE)
    log_message(f"Found {len(processed_paths)} already processed paths to skip.")  
    paths_to_process = [p for p in input_paths if p not in processed_paths]
    log_message(f"Total paths: {len(input_paths)} | To process: {len(paths_to_process)}")
    log_message(f"Starting concurrent processing with {MAX_WORKERS} workers...")
    main_client_cycle = cycle(main_clients)
    tool_client_cycle = cycle(tool_clients)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_path, path, as_info_db, ground_truth_db, next(main_client_cycle), next(tool_client_cycle))
            for path in paths_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(paths_to_process), desc="Processing AS Paths"):
            try:
                future.result()  
            except Exception as exc:
                log_message(f'A path generated an exception: {exc}')

    log_message("===== Script finished. All paths processed. =====")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_message(f"A fatal error occurred in main execution: {e}")
        import traceback
        log_message(traceback.format_exc())
