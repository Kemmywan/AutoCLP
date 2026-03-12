import json

class RecoveryAdvice:
    def __init__(self, data: dict={}):
        self.data = data or {}
    
    @classmethod
    async def from_diagnostic(cls, diag: dict, agent):
        res = await agent.run(task=json.dumps(diag))
        return cls(json.loads(res.messages[1].to_text()))

    def to_json(self):
        return json.dumps(self.data, ensure_ascii=False, indent=4)