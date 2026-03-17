from core import *
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import asyncio
import argparse
import random
from matplotlib.font_manager import FontProperties


my_font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

async def exp_record(dialogue: str):

    llms = LLMManager()

    start = time.time()
    medical_record = await MedicalRecord.from_dialogue(dialogue, llms.agent_1)
    end = time.time()

    time_cost = end - start
    return time_cost, medical_record.to_json()

async def main() -> None:

    parser = argparse.ArgumentParser(description="Test for record")

    parser.add_argument(
        "-t", "--total",
        type=int,
        required=True,
        help="每组采样"
    )

    args = parser.parse_args()

    times_llm_total = []
    times_person_total = []

    for i in range(3, 26):
        with open(f"test_data/2020_{i*100}_{(i+1)*100}_20.txt") as f:
            dialogues = json.loads(f.read())
            times_llm = []
            times_person = []
            for _ in range(args.total):
                while True:
                    item = dialogues[random.randint(0, len(dialogues) - 1)]

                    time, record = await exp_record(item["dialogue"])

                    if record != {}:
                        break

                times_llm.append(time)
                times_person.append(len(record)) #大约60个字每分钟

                print(f"Finish {_}/{args.total} for 2020_{i*100}_{(i+1)*100}_20.txt.")

            times_llm_total.append(sum(times_llm) / len(times_llm))
            times_person_total.append(sum(times_person) / len(times_person))

    times_llm_np = np.array(times_llm_total)
    times_person_np = np.array(times_person_total)

    np.savez('exp_result/bench_results.npz', llm = times_llm_total,  person=times_person_total)

    range_labels = [f'{i}~{(i+1)}' for i in range(3, 26)]
    # range_labels = ['0~100', '100~200', '200~300', '300~400', '400~500', '500~600', '600~700', '700~800', '800~900', '900~1000', '1000~1100', '1100~1200', '1200~1300', '1300~1400', '1400~1500', '1500~1600', '1600~1700', '1700~1800', '1800~1900', '1900~2000', '2000~2100', '2100~2200', '2200~2300', '2300~2400', '2400~2500', '2500~2600']

    x = np.arange(len(range_labels))

    plt.figure(figsize=(13, 5))
    plt.plot(times_llm_np, label='LLM病历耗时', marker='o')
    plt.plot(times_person_np, label='人工病历耗时', marker='s')


    plt.legend(prop=my_font)
    plt.xticks(ticks=x, labels=range_labels)
    plt.xlabel('对话长度（100字）', fontproperties=my_font)
    plt.ylabel('耗时（秒）', fontproperties=my_font)
    plt.title('LLM vs 人工病历书写耗时对比折线图', fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('exp_result/test_record.png')
    plt.show()
    
if __name__ == "__main__":
    asyncio.run(main())        
