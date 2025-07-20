#!/usr/bin/env python
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
# ===== 配置区域 =====
MAIN_LLM_URLS = [f"http://localhost:{port}/v1" for port in [8000, 8001, 8002, 8003]]
LOCAL_MODEL_NAME = "qwen2.5-7b"
MODES_TO_RUN = ['org', 'base']
MAX_WORKERS = 12
BATCH_SAVE_SIZE = 100
MAX_TOKENS_OUTPUT = 30000
INPUT_PATHS_TXT_FILE = "/home/u2021110842/autodl-tmp/BGPAgent/20250211data/new_dataset.txt"
AS_INFO_DB_FILE = "/home/u2021110842/autodl-tmp/BGPAgent/20250211data/asn_organization_mapping.json"
date_str = datetime.today().strftime('%Y%m%d')
output_dir = f"inference_results_{date_str}"
os.makedirs(output_dir, exist_ok=True)
LOG_FILE = os.path.join(output_dir, f"processing_log_{date_str}.txt")
ENABLE_DEBUG_OUTPUTS = True
if ENABLE_DEBUG_OUTPUTS:
    DEBUG_DIR = f"debug_logs_{date_str}_inference"
    os.makedirs(DEBUG_DIR, exist_ok=True)


# ===== 新增：为 'org' 模式设计的 Few-Shot Prompt 模板 =====
FEW_SHOT_PROMPT_TEMPLATE_ORG = """You are an expert BGP analysis model. Your sole task is to analyze the provided AS Path and organization names, then output the business relationships as a single, valid JSON list of strings. Do not add any explanation or any other text.

---
User: Please use the given as path, knowned AS organization name (orgName) infer business relationships:.As path: 2914|58453|9808. ASes' OrgName: 2914's OrgName is NTT Communications Corporation
58453's OrgName is China Mobile International Limited
9808's OrgName is China Mobile Guangdong. ASes' Relationship ASes' Relationship is
Assistant: ["2914-58453:p2p", "58453-9808:p2c"]
---
User: Please use the given as path, knowned AS organization name (orgName) infer business relationships:.As path: 20514|1299|3356|209|36811. ASes' OrgName: 20514's OrgName is Axians AB
1299's OrgName is Arelion Sweden AB
3356's OrgName is Level 3 Parent, LLC
209's OrgName is CenturyLink Communications, LLC
36811's OrgName is Wiztech Internet. ASes' Relationship ASes' Relationship is
Assistant: ["20514-1299:c2p", "1299-3356:p2p", "3356-209:p2p", "209-36811:p2c"]
---
User: {FINAL_QUESTION_PLACEHOLDER}
Assistant:
"""

# ===== 新增：为 'base' 模式设计的 Few-Shot Prompt 模板 =====
FEW_SHOT_PROMPT_TEMPLATE_BASE = """You are an expert BGP analysis model. Your sole task is to analyze the provided AS Path and infer the business relationships as a single, valid JSON list of strings. You must make your best guess based on common BGP routing policies and typical AS numbers, without organization names. Do not add any explanation or any other text.

---
User: Please use the given as path to infer business relationships:.As path: 64512|64513|64514. ASes' Relationship ASes' Relationship is
Assistant: ["64512-64513:c2p", "64513-64514:c2p"]
---
User: Please use the given as path to infer business relationships:.As path: 3356|1299|2914. ASes' Relationship ASes' Relationship is
Assistant: ["3356-1299:p2p", "1299-2914:p2p"]
---
User: {FINAL_QUESTION_PLACEHOLDER}
Assistant:
"""

# ===== 全局锁 =====
log_lock = threading.Lock()
file_write_lock = threading.Lock()


def extract_json_list_from_string(text):
    """
    根据用户观察进行优化的新版提取函数。
    1. 使用非贪婪模式(r'\[.*?\]')来查找所有可能的JSON列表。
    2. 返回最后一个匹配到的结果，因为总结通常出现在末尾。
    """
    # 使用非贪婪模式(r'\[.*?\]')来查找所有可能的JSON列表
    # re.DOTALL 确保了可以匹配包含换行的列表
    matches = re.findall(r'\[.*?\]', text, re.DOTALL)

    # 根据您的观察，优先选择最后一个匹配项
    if matches:
        return matches[-1] # 返回最后一个找到的列表字符串

    # 如果没有找到任何匹配项，返回一个空列表字符串
    return "[]"

def log_message(message):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(log_entry + "\n")

def load_json_file(filename):
    if not os.path.exists(filename):
        log_message(f"严重错误: 文件 {filename} 不存在。程序将退出。")
        exit(1)
    try:
        with open(filename, "r", encoding="utf-8") as f: return json.load(f)
    except json.JSONDecodeError as e:
        log_message(f"严重错误: JSON文件 {filename} 格式错误: {e}")
        exit(1)
    return {}

def load_processed_paths(json_file):
    seen = set()
    if not os.path.exists(json_file): return seen
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item_list in data:
                if not item_list: continue
                entry = item_list[0]
                question = entry.get("question", "")
                match = re.search(r'As path: (.*?)\.', question)
                if match: seen.add(match.group(1).strip())
    except (json.JSONDecodeError, IndexError, TypeError):
        log_message(f"警告: 无法正确解析已有的结果文件 {json_file}。将视为空文件处理。")
    return seen

def save_debug_log(as_path, debug_data):
    if not ENABLE_DEBUG_OUTPUTS: return
    fname = as_path.replace('|', '_').replace('/', '_') + '.json'
    with open(os.path.join(DEBUG_DIR, fname), 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)

def get_llm_response(client, model_name, prompt, temperature=0.01):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS_OUTPUT)
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"LLM调用失败 at {client.base_url} with model {model_name}: {e}")
        return None

def append_results_to_json_file(file_path, results_batch):
    with file_write_lock:
        log_message(f"触发批量保存，正在将 {len(results_batch)} 条结果写入 {file_path}...")
        existing_data = []
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: existing_data = json.load(f)
            except json.JSONDecodeError:
                log_message(f"警告: {file_path} 文件格式损坏，将创建新文件。")
                existing_data = []
        
        existing_data.extend([[result] for result in results_batch])
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(existing_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            log_message(f"!!! 文件批量保存失败: {e}")

# ===== 修改: Prompt构建函数使用新模板 =====
def build_prompt_for_mode(as_path, as_info_db, mode):
    if mode == 'org':
        # 1. 构建 'org' 模式的独特问题字符串
        question_part = f"Please use the given as path, knowned AS organization name (orgName) infer business relationships:.As path: {as_path}. "
        asn_list = [x.strip() for x in as_path.split("|") if x.strip()]
        org_info_parts = [f"{asn}'s OrgName is {as_info_db.get(asn) or 'Unknown'}" for asn in asn_list]
        question_part += f"ASes' OrgName: {'\n'.join(org_info_parts)}. "
        question_part += "ASes' Relationship ASes' Relationship is"
        
        # 2. 将问题字符串注入 'org' 模板
        full_prompt = FEW_SHOT_PROMPT_TEMPLATE_ORG.replace("{FINAL_QUESTION_PLACEHOLDER}", question_part)
    
    else: # base mode
        # 1. 构建 'base' 模式的独特问题字符串 (更简洁)
        question_part = f"Please use the given as path to infer business relationships:.As path: {as_path}. ASes' Relationship ASes' Relationship is"
        
        # 2. 将问题字符串注入 'base' 模板
        full_prompt = FEW_SHOT_PROMPT_TEMPLATE_BASE.replace("{FINAL_QUESTION_PLACEHOLDER}", question_part)

    return full_prompt, question_part

def process_path(as_path, as_info_db, client_main, mode, model_name):
    """处理单个AS路径的核心函数"""
    debug_log = {'as_path': as_path, 'mode': mode}
    full_prompt, question_to_log = build_prompt_for_mode(as_path, as_info_db, mode)
    debug_log['full_prompt'] = full_prompt

    raw_output = get_llm_response(client_main, model_name, full_prompt)
    if not raw_output:
        log_message(f"失败: LLM调用失败，路径: '{as_path}'.")
        debug_log['outcome'] = 'Failed (LLM call failed)'
        save_debug_log(as_path, debug_log)
        return None

    extracted_list_str = extract_json_list_from_string(raw_output)
    result_entry = {
        "question": question_to_log,
        f"{model_name}-answer": raw_output,
        f"{model_name}-answer-list": [extracted_list_str]
    }
    
    debug_log['outcome'] = 'Success (Generated)'
    debug_log['parsed_result'] = result_entry
    save_debug_log(as_path, debug_log)
    return result_entry


def main():
    log_message("===== 全自动多模式脚本启动 (双模板Few-shot增强版 V2) =====")

    main_clients = [openai.OpenAI(api_key="none", base_url=url, timeout=300.0) for url in MAIN_LLM_URLS]
    as_info_db = load_json_file(AS_INFO_DB_FILE)
    
    if os.path.exists(INPUT_PATHS_TXT_FILE):
        with open(INPUT_PATHS_TXT_FILE, 'r', encoding='utf-8') as f:
            input_paths = ["|".join(line.strip().split()) for line in f if line.strip()]
    else:
        log_message(f"严重错误: 输入文件 {INPUT_PATHS_TXT_FILE} 不存在。程序退出。")
        return

    log_message(f"成功从输入文件加载 {len(input_paths)} 条总路径。")

    for mode in MODES_TO_RUN:
        log_message("-" * 60 + f"\n===== 开始处理模式: [{mode}] =====" + "-" * 60)
        
        output_filename = f"inference_results_{date_str}_{LOCAL_MODEL_NAME}-{mode}.json"
        FINAL_OUTPUT_FILE = os.path.join(output_dir, output_filename)
        log_message(f"当前模式的输出文件为: {FINAL_OUTPUT_FILE}")

        processed_paths = load_processed_paths(FINAL_OUTPUT_FILE)
        paths_to_process = [p for p in input_paths if p not in processed_paths]
        log_message(f"模式 '{mode}' 待处理路径数: {len(paths_to_process)}")
        
        if not paths_to_process:
            log_message(f"模式 '{mode}' 没有新的路径需要处理，跳至下个模式。")
            continue

        main_client_cycle = cycle(main_clients)
        batch_results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_path, path, as_info_db, next(main_client_cycle), mode, LOCAL_MODEL_NAME): path for path in paths_to_process}
            
            for future in tqdm(as_completed(futures), total=len(paths_to_process), desc=f"处理AS路径 (模式: {mode})"):
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        if len(batch_results) >= BATCH_SAVE_SIZE:
                            append_results_to_json_file(FINAL_OUTPUT_FILE, batch_results)
                            batch_results = []
                except Exception as exc:
                    path = futures[future]
                    log_message(f'路径 {path} 在模式 {mode} 处理时产生异常: {exc}')

        if batch_results:
            log_message("正在保存最后一批不足100条的结果...")
            append_results_to_json_file(FINAL_OUTPUT_FILE, batch_results)
        
        log_message(f"===== 模式 [{mode}] 处理完毕 =====")

    log_message("=" * 60 + "\n===== 脚本结束。所有模式处理完毕。 =====")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_message(f"主程序执行时发生致命错误: {e}")
        import traceback
        log_message(traceback.format_exc())
