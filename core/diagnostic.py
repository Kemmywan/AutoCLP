import json

class Diagnostic:
    def __init__(self, data: dict={}):
        self.data = data or {}
    
    @classmethod
    async def from_exam_and_record(cls, exam: dict, record: dict, agent, rag_context=''):

        if rag_context:
            task = (
                f"以下是历史相似病例，供参考：\n"
                f"{rag_context}\n\n"
                f"---\n"
                f"请根据以下当前病历和检查结果给出诊断结论：\n"
                f"病历记录：{json.dumps(record, ensure_ascii=False)}\n"
                f"检查结果：{json.dumps(exam, ensure_ascii=False)}"
            )
        else:
            # 无历史参考时，维持原逻辑不变，冷启动
            task = json.dumps(exam, ensure_ascii=False) + json.dumps(record, ensure_ascii=False)

        res = await agent.run(task=task)
        return cls(json.loads(res.messages[1].to_text()))

    def to_json(self):
        return json.dumps(self.data, ensure_ascii=False, indent=4)
    