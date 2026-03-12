import json

class MedicalRecord:
    def __init__(self, data: dict = {}):
        self.data = data or {}

    @classmethod
    async def from_dialogue(cls, dialogue: str, agent):
        # Use llm to package dialogue
        res = await agent.run(task=dialogue)
        return cls(json.loads(res.messages[1].to_text()))
    
    def to_json(self):
        return json.dumps(self.data, ensure_ascii=False, indent=4)
    
