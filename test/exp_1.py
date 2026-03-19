"""
Experiment 1: 对 data_for_exp_1.txt 中的 60 个样本逐一执行完整流程，
记录四个阶段耗时并保存结果。

四个阶段:
  1. RawClinicalData 生成耗时
  2. CommanderLLM 生成 TaskList 耗时
  3. AutoCPL 生成 + 翻译(interpret) 总耗时
  4. LLMPool 依次完成 task 耗时

用法: python test/exp_1.py
"""

import sys
import os
import json
import time
import asyncio
import numpy as np

# 将项目根目录加入 sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from ambient import MultimodalAdapter
from commander import CommanderLLM
from cpl import CPLGenerator, CPLInterpreter
from llm_manager import LLMManager

DATA_PATH = os.path.join(ROOT_DIR, "test_data", "data_for_exp_1.txt")
EXP_RESULT_DIR = os.path.join(ROOT_DIR, "exp_result")


async def run_single_sample(dialogue_text: str, adapter, commander, generator, interpreter, llm_mgr):
    """
    执行单个样本的完整流程，返回四个阶段的耗时 (秒)。
    """
    # 阶段1: RawClinicalData 生成
    t0 = time.perf_counter()
    raw_data = adapter.ingest_from_string(dialogue_text)
    t1 = time.perf_counter()

    # 阶段2: Commander LLM 生成 TaskList
    tasks = await commander.decompose(raw_data)
    t2 = time.perf_counter()

    # 阶段3: AutoCPL 生成 + 翻译
    script = generator.generate(tasks, pathway_name="门诊处理路径")
    plan = interpreter.interpret_script(script)
    t3 = time.perf_counter()

    # 阶段4: LLMPool 依次完成 task
    await llm_mgr._pool.execute_plan(
        plan,
        context={"dialogue": raw_data.content}
    )
    t4 = time.perf_counter()

    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3)


async def main():
    # 加载数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    n = len(samples)
    print(f"共加载 {n} 个样本，开始实验...\n")

    # 初始化组件（只初始化一次）
    adapter = MultimodalAdapter()
    llm_mgr = LLMManager()
    commander = CommanderLLM(agent=llm_mgr.commander_agent)
    generator = CPLGenerator()
    interpreter = CPLInterpreter()

    timings = np.zeros((n, 4), dtype=np.float64)

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", "?")
        source = sample.get("source_file", "?")
        print(f"[{i + 1}/{n}] 样本 id={sample_id} (from {source}) ...", end=" ", flush=True)

        try:
            costs = await run_single_sample(
                sample["dialogue"], adapter, commander, generator, interpreter, llm_mgr
            )
            timings[i] = costs
            print(f"完成  raw={costs[0]:.3f}s  cmd={costs[1]:.3f}s  cpl={costs[2]:.3f}s  pool={costs[3]:.3f}s")
        except Exception as e:
            print(f"失败: {e}")
            timings[i] = [float("nan")] * 4

    # 计算平均耗时（忽略 NaN）
    avg_timings = np.nanmean(timings, axis=0, keepdims=True)  # (1, 4)

    # 保存结果
    os.makedirs(EXP_RESULT_DIR, exist_ok=True)
    result_path = os.path.join(EXP_RESULT_DIR, "exp1_results.npz")
    np.savez(result_path, timings=timings, avg_timings=avg_timings)
    print(f"\n结果已保存 → {result_path}")

    # 输出平均耗时
    labels = ["RawClinicalData生成", "Commander生成TaskList", "AutoCPL生成+翻译", "LLMPool执行Task"]
    print("\n========== 平均耗时 ==========")
    for label, val in zip(labels, avg_timings[0]):
        print(f"  {label:<22s}: {val:.4f} s")
    print(f"  {'总计':<22s}: {avg_timings[0].sum():.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
