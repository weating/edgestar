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
import argparse
import ast
import sys
from collections import Counter
from typing import Dict, Any, Set, Tuple

# ==============================================================================
# ===== Part 1: 配置与全局辅助函数 ============================================
# ==============================================================================
# (此部分无变化)
def log_message(message: str):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)

def load_json_file(filepath: str, file_desc: str) -> Dict:
    log_message(f"🔄 正在加载 {file_desc} 文件: {filepath}")
    if not os.path.exists(filepath):
        log_message(f"  ❌ 错误: 文件未找到 '{filepath}'。程序将退出。")
        sys.exit(1)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            log_message(f"  ✅ 文件加载成功，包含 {len(data)} 条记录。")
            return data
    except json.JSONDecodeError:
        log_message(f"  ❌ 错误: 解析JSON文件 '{filepath}' 失败。程序将退出。")
        sys.exit(1)
        
def append_to_json_file(filepath: str, batch_data: list):
    """线程安全地将一批数据追加到JSON文件中。"""
    with file_write_lock:
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
        except json.JSONDecodeError:
            log_message(f"警告: JSON文件 '{filepath}' 格式损坏，将创建新文件。")
            existing_data = []
        
        existing_data.extend(batch_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

# ==============================================================================
# ===== Part 2: 核心LLM推理逻辑 ================================================
# ==============================================================================
# (此部分无变化)
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

def build_prompt_for_mode(as_path: str, as_info_db: Dict, mode: str) -> Tuple[str, str]:
    if mode == 'org':
        question_part = f"Please use the given as path, knowned AS organization name (orgName) infer business relationships:.As path: {as_path}. "
        asn_list = [x.strip() for x in as_path.split("|") if x.strip()]
        org_info_parts = [f"{asn}'s OrgName is {as_info_db.get(asn) or 'Unknown'}" for asn in asn_list]
        question_part += f"ASes' OrgName: {'\n'.join(org_info_parts)}. ASes' Relationship ASes' Relationship is"
        full_prompt = FEW_SHOT_PROMPT_TEMPLATE_ORG.replace("{FINAL_QUESTION_PLACEHOLDER}", question_part)
    else:
        question_part = f"Please use the given as path to infer business relationships:.As path: {as_path}. ASes' Relationship ASes' Relationship is"
        full_prompt = FEW_SHOT_PROMPT_TEMPLATE_BASE.replace("{FINAL_QUESTION_PLACEHOLDER}", question_part)
    return full_prompt, question_part

def get_llm_response(client: openai.OpenAI, model_name: str, prompt: str) -> str | None:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01, max_tokens=8192)
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"LLM调用失败 at {client.base_url} with model {model_name}: {e}")
        return None

def process_path_inference(as_path: str, as_info_db: Dict, client: openai.OpenAI, mode: str, model_name: str) -> Dict | None:
    full_prompt, question_to_log = build_prompt_for_mode(as_path, as_info_db, mode)
    raw_output = get_llm_response(client, model_name, full_prompt)
    if not raw_output:
        log_message(f"失败: LLM调用失败，路径: '{as_path}'.")
        return None
    matches = re.findall(r'\[.*?\]', raw_output, re.DOTALL)
    extracted_list_str = matches[-1] if matches else "[]"
    return {"question": question_to_log, f"{model_name}-answer": raw_output, f"{model_name}-answer-list": [extracted_list_str]}

def run_inference_phase(args: argparse.Namespace, as_info_db: Dict, modes: list) -> Dict[str, str]:
    log_message("--- [阶段 1/3] 开始推理任务 ---")
    main_client = openai.OpenAI(api_key="none", base_url=f"http://localhost:{args.port}/v1", timeout=300.0)
    with open(args.bgp_paths_input, 'r', encoding='utf-8') as f:
        input_paths = ["|".join(line.strip().split()) for line in f if line.strip()]
    if args.limit and args.limit > 0:
        log_message(f"--- ⚠️  测试模式启动：仅处理前 {args.limit} 条路径 ---")
        input_paths = input_paths[:args.limit]
    log_message(f"成功从输入文件加载 {len(input_paths)} 条总路径进行处理。")
    output_files = {}
    for mode in modes:
        log_message(f"===== 开始处理模式: [{mode}] =====")
        output_filename = os.path.join(args.output_dir, f"inference_results_{datetime.today().strftime('%Y%m%d')}_{args.model_name}-{mode}.json")
        output_files[mode] = output_filename
        all_results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_path_inference, path, as_info_db, main_client, mode, args.model_name): path for path in input_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"推理模式: {mode}"):
                result = future.result()
                if result:
                    all_results.append([result])
        log_message(f"正在保存模式 '{mode}' 的原始推理结果到: {output_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
    log_message("--- [阶段 1/3] 推理任务全部完成 ---")
    return output_files

# ==============================================================================
# ===== Part 3: 结果整合与投票逻辑 ============================================
# ==============================================================================
# (与上一回复相同)
def run_voting_phase(raw_result_file: str, voted_result_file: str, model_name: str):
    log_message(f"--- [阶段 2/3] 开始对文件 '{os.path.basename(raw_result_file)}' 进行投票处理 ---")
    data = load_json_file(raw_result_file, "LLM原始结果")
    answer_key = f"{model_name}-answer-list"
    all_relations = {}
    for item_list in data:
        for item in item_list:
            try:
                relations_list = ast.literal_eval(item.get(answer_key, ["[]"])[0])
                for rel_str in relations_list:
                    match = re.match(r'(\d+)-(\d+):(\w+)', rel_str.strip())
                    if not match: continue
                    as1, as2, rel_type = int(match.group(1)), int(match.group(2)), match.group(3)
                    if as1 < as2:
                        key, std_rel = f"{as1}-{as2}", rel_type
                    else:
                        key = f"{as2}-{as1}"
                        std_rel = {'c2p': 'p2c', 'p2c': 'c2p'}.get(rel_type, rel_type)
                    if key not in all_relations: all_relations[key] = []
                    all_relations[key].append(std_rel)
            except (ValueError, SyntaxError): continue
    final_relations_bidirectional = {}
    for key, rel_list in all_relations.items():
        as1_str, as2_str = key.split('-')
        majority_vote = Counter(rel_list).most_common(1)[0][0]
        final_relations_bidirectional[f"{as1_str}-{as2_str}"] = majority_vote
        reversed_vote = {'p2c': 'c2p', 'c2p': 'p2c'}.get(majority_vote, majority_vote)
        final_relations_bidirectional[f"{as2_str}-{as1_str}"] = reversed_vote
    log_message(f"投票处理完成，正在双向保存整合后结果到: {voted_result_file}")
    with open(voted_result_file, 'w', encoding='utf-8') as f:
        json.dump(final_relations_bidirectional, f, indent=2, ensure_ascii=False)
    log_message(f"--- [阶段 2/3] 投票处理完成 ---")

# ==============================================================================
# ===== Part 4: 最终评测与报告逻辑 (最终版) ====================================
# ==============================================================================
def run_evaluation_phase(args: argparse.Namespace, mode: str, voted_llm_file: str):
    """执行最终的详细评测并生成报告。"""
    log_message(f"--- [阶段 3/3] 开始对模式 '{mode}' 进行最终评测 ---")
    
    with open(args.bgp_paths_input, 'r') as f:
        pairs_to_check = {tuple(sorted((p[0], p[1]))) for line in f for p in zip(line.strip().split(), line.strip().split()[1:]) if p[0].isdigit() and p[1].isdigit() and p[0] != p[1]}

    ground_truth = load_json_file(args.ground_truth, "Ground-Truth")
    llm_results = load_json_file(voted_llm_file, "LLM整合结果")
    
    correct_count, wrong_count, not_present_count = 0, 0, 0
    tp, fp, fn = [0, 0], [0, 0], [0, 0]
    
    def get_class_idx(rel_type):
        if rel_type == 'p2p': return 0
        if rel_type in ['p2c', 'c2p']: return 1
        return -1

    for as1, as2 in pairs_to_check:
        key = f"{as1}-{as2}"
        gt_value = ground_truth.get(key)
        
        if gt_value is None: continue

        llm_value = llm_results.get(key)
        
        if llm_value is None:
            not_present_count += 1
        elif llm_value == gt_value:
            correct_count += 1
        else:
            wrong_count += 1

        gt_class_idx, llm_class_idx = get_class_idx(gt_value), get_class_idx(llm_value)
        if gt_class_idx != -1:
            if llm_class_idx == gt_class_idx:
                tp[gt_class_idx] += 1
            else:
                fn[gt_class_idx] += 1
                if llm_class_idx != -1:
                    fp[llm_class_idx] += 1

    total_evaluated = correct_count + wrong_count + not_present_count
    total_accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    
    def calculate_metrics(tp_val, fp_val, fn_val):
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        return precision, recall

    p2p_precision, p2p_recall = calculate_metrics(tp[0], fp[0], fn[0])
    transit_precision, transit_recall = calculate_metrics(tp[1], fp[1], fn[1])

    report_path = os.path.join(args.output_dir, f"llm_performance_report_{args.model_name}-{mode}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"      LLM 性能评估报告 ({mode.upper()} 模式)\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型名称: {args.model_name}\n\n")
        f.write("--- 总体准确率 (Overall Accuracy) ---\n")
        f.write(f"评测范围: 在 Ground-Truth 中存在的 AS 对总数\n")
        f.write(f"  - 分母 (实际评测数): {total_evaluated}\n")
        f.write(f"  - 分子 (正确预测数): {correct_count}\n")
        f.write(f"  - (其中LLM未提供结果数: {not_present_count})\n")
        f.write(f"总准确率 (ACC) = 正确数 / 实际评测数 = {total_accuracy:.2%}\n\n")
        f.write("--- 分类性能指标 (Precision & Recall) ---\n")
        f.write(f"{'类别 (Class)':<12} | {'精确率 (Precision)':<20} | {'召回率 (Recall)':<20}\n")
        f.write(f"{'-'*12}-+-{'-'*20}-+-{'-'*20}\n")
        f.write(f"{'p2p':<12} | {p2p_precision:<20.4f} | {p2p_recall:<20.4f}\n")
        f.write(f"{'p2c/c2p':<12} | {transit_precision:<20.4f} | {transit_recall:<20.4f}\n")
        f.write("\n" + "="*60 + "\n")
    log_message(f"✅ 评测报告已生成: {report_path}")

# ==============================================================================
# ===== Part 5: 主流程协调器 ===================================================
# ==============================================================================
def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    as_info_db = load_json_file(args.as_org_db, "AS组织信息")
    modes_to_run = ['org', 'base']
    raw_output_files = run_inference_phase(args, as_info_db, modes_to_run)
    for mode in modes_to_run:
        log_message("="*80)
        log_message(f"开始对【{mode.upper()}】模式的结果进行后处理与评测")
        raw_file = raw_output_files.get(mode)
        if not raw_file or not os.path.exists(raw_file):
            log_message(f"⚠️ 警告: 未找到模式 '{mode}' 的推理结果文件。跳过此模式的后处理。")
            continue
        voted_file = raw_file.replace('.json', '-voted.json')
        run_voting_phase(raw_file, voted_file, args.model_name)
        run_evaluation_phase(args, mode, voted_file)
    log_message("="*80)
    log_message("🎉🎉🎉 全流程处理完毕！")
    log_message(f"所有中间文件和最终报告均已保存在目录: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM BGP关系推理、整合与评测全流程统一脚本。")
    parser.add_argument("--port", type=int, default=8000, help="VLLM服务运行的端口号。")
    parser.add_argument("--model-name", type=str, required=True, help="由VLLM服务暴露的模型名称。")
    parser.add_argument("--max-workers", type=int, default=12, help="推理时使用的最大并发线程数。")
    parser.add_argument("--limit", type=int, default=None, help="仅处理指定数量的输入路径以进行快速测试。")
    parser.add_argument("--bgp-paths-input", required=True, help="BGP路径输入文件(.txt)。")
    parser.add_argument("--as-org-db", required=True, help="AS组织名称映射文件(.json)。")
    parser.add_argument("--ground-truth", required=True, help="Ground-Truth关系文件(.json)。")
    parser.add_argument("--output-dir", default=f"llm_pipeline_results_{datetime.today().strftime('%Y%m%d')}", help="所有输出文件（中间结果、报告）的根目录。")
    args = parser.parse_args()
    main(args)