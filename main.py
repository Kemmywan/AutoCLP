from core import *
import argparse
import asyncio
import time
import json
from rag import rag_core

vector_memory = rag_core.VectorMemory()

async def run_diagnostic_with_rag(
    exam_result: ExamResult,
    medical_record: MedicalRecord,
    agent
):
    print("Agent is working out the diagnostic...")

    query = f"{medical_record.data} {exam_result.data}"
    rag_results = vector_memory.search(query, top_k=3)

    if rag_results:
        rag_context = "\n\n".join([
            f"[历史参考案例 {i+1}]\n"
            f"病历记录: {r.get('record', '')}\n"
            f"检查结果: {r.get('exam_result', '')}\n"
            f"诊断结论: {r.get('diagnostic', '')}"
            for i, r in enumerate(rag_results)
        ])
        print(f"[RAG] 召回 {len(rag_results)} 条相似历史案例，注入诊断上下文。")
    else:
        rag_context = "（暂无历史参考案例）"
        print("[RAG] 本地数据库暂无相似案例，冷启动诊断。")

    diagnostic = await Diagnostic.from_exam_and_record(
        exam_result.data,
        medical_record.data,
        agent,
        rag_context=rag_context  # 需在Diagnostic方法中接收此参数
    )

    print(f"\nThe diagnostic:\n{diagnostic.to_json()}\n")
    print("=" * 15)

    print("[用户接口] 是否接受以上诊断结论？")
    print("  [Enter] 直接确认")
    print("  [输入内容] 手动覆写诊断结论")

    user_input = input(">>> ").strip()

    if user_input:
        # 用户手动覆写
        final_diagnostic = user_input
        print(f"[已覆写] 最终诊断结论：{final_diagnostic}")
    else:
        # 用户接受AI诊断
        final_diagnostic = diagnostic.data  # 取diagnostic对象中的文本字段
        print(f"[已确认] 最终诊断结论：{final_diagnostic}")

    triple = {
        'record': medical_record.data,
        'exam_result': exam_result.data,
        'diagnostic': final_diagnostic
    }
    vector_memory.add_triple(triple)
    vector_memory.save_index()  # 持久化embedding到磁盘
    print("[RAG] 本次诊断已归档至本地数据库。")

    import time
    time.sleep(0.5)

    return Diagnostic(json.loads(final_diagnostic) if isinstance(final_diagnostic, str) else final_diagnostic)



async def main() -> None:

    parser = argparse.ArgumentParser(description="AutoCLP")
    parser.add_argument(
          "-d", "--dialogue",
          type=str,
          required=True,
          help="对话输入文件（txt/json等，内容为医患对话）"
    )
    args = parser.parse_args()

    with open(args.dialogue, "r", encoding="utf-8") as f:
        dialogue=f.read()

    print("Agent starting...\n")
    time.sleep(0.5)

    llms = LLMManager()

    print("Agent is making a medical record...")
    medical_record = await MedicalRecord.from_dialogue(dialogue, llms.agent_1)
    print(f"\nThe medical record:\n\n{medical_record.to_json()}\n")
    print("="*15)
    time.sleep(0.5)

    print("Agent is offering a schedule...")
    schedule = await Schedule.from_medical_record(medical_record.data, llms.agent_2)
    print(f"\nThe schedule for patient:\n{schedule.to_json()}\n")
    print("="*15)
    time.sleep(0.5)

    print("Agent is generating exam result...(Test Only)")
    exam_result = await ExamResult.generate(schedule.data, medical_record.data, llms.agent_test)
    print(f"\nThe generated result is:\n{exam_result.to_json()}\n")
    print("="*15)
    time.sleep(0.5)

    diagnostic = await run_diagnostic_with_rag(
        exam_result,
        medical_record,
        llms.agent_3
    )
    # print("Agent is working out a diagnostic...")
    # diagnostic = await Diagnostic.from_exam_and_record(exam_result.data, medical_record.data, llms.agent_3)
    # print(f"\nThe diagnostic:\n{diagnostic.to_json()}\n")
    # print("="*15)
    # time.sleep(0.5)

    print("Agent is offering recovery advice...")
    recovery_advice = await RecoveryAdvice.from_diagnostic(diagnostic.data, llms.agent_4)
    print(f"\nThe recovery advice includes:\n{recovery_advice.to_json()}\n")
    time.sleep(0.5)
    print("="*15)

    print("Thanks for using our clincal agent!")

if __name__ == "__main__":
    asyncio.run(main())